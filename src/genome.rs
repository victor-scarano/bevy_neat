use std::{cmp, collections::{BTreeMap, BTreeSet}, fmt::Debug, iter, ops::{AddAssign, Sub}, sync::Arc};
use crate::{config, conn, node, traits::{self, Activation, Config, ConnGene}};
use bevy::ecs::component::Component;
use rand::{seq::{IteratorRandom, SliceRandom}, Rng};


#[derive(Clone, Component, Debug, PartialEq)]
pub struct FeedForwardGenome {
    conn_genes: BTreeSet<conn::ConnGene<Self>>,
    input_nodes: Arc<[node::NodeGene<Self>]>,
    hidden_nodes: BTreeSet<node::NodeGene<Self>>,
    output_nodes: Arc<[node::NodeGene<Self>]>,
    fitness: Option<f32>,
}

impl FeedForwardGenome {
    fn matching_genes<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = (&conn::ConnGene<Self>, &conn::ConnGene<Self>)> + 'a {
        self.conn_genes.intersection(&other.conn_genes).map(|conn|
            (self.conn_genes.get(conn).unwrap(), other.conn_genes.get(conn).unwrap())
        )
    }

    fn disjoint_genes<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &conn::ConnGene<Self>> + 'a {
        let min_innov = cmp::min(self.conn_genes.first().unwrap().innov(), other.conn_genes.first().unwrap().innov());
        self.conn_genes.symmetric_difference(&other.conn_genes).filter(move |conn| conn.innov().le(&min_innov))
    }

    fn excess_genes<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &conn::ConnGene<Self>> + 'a {
        let min_innov = cmp::min(
            self.conn_genes.first().unwrap().innov(),
            other.conn_genes.first().unwrap().innov()
        );

        self.conn_genes.symmetric_difference(&other.conn_genes).collect::<Vec<_>>()
            .into_iter().rev().take_while(move |conn| conn.innov().gt(&min_innov))
    }
}

impl traits::Genome for FeedForwardGenome {
    type Config = config::Config<Self>;
    type ConnGene = conn::ConnGene<Self>;
    type NodeGene = node::NodeGene<Self>;

    fn minimal(config: &Self::Config) -> Self {
        Self {
            conn_genes: BTreeSet::new(),
            input_nodes: iter::repeat(node::NodeGene::new_input()).take(config.input_len()).collect(),
            hidden_nodes: BTreeSet::new(),
            output_nodes: iter::repeat(node::NodeGene::new_output()).take(config.output_len()).collect(),
            fitness: None,
        }
    }

    fn add_conn_mut(&mut self, config: &Self::Config) -> Self::ConnGene {
        let mut possible_in_nodes = self.input_nodes.iter().chain(&self.hidden_nodes).cloned().collect::<Vec<_>>();
        assert_ne!(possible_in_nodes.len(), 0);
        possible_in_nodes.shuffle(&mut rand::thread_rng());

        let mut possible_out_nodes = self.output_nodes.iter().chain(&self.hidden_nodes).cloned().collect::<Vec<_>>();
        assert_ne!(possible_out_nodes.len(), 0);
        possible_out_nodes.shuffle(&mut rand::thread_rng());

        let in_node = possible_in_nodes.iter().find(|node| possible_out_nodes.iter().count()
            .saturating_sub(self.hidden_nodes.contains(node) as usize)
            .saturating_sub(node.forward(|forward| forward.len())).gt(&0)
        ).unwrap();
        
        let out_node = possible_out_nodes.iter().find(|node|
            node.backward(|backward| !backward.iter().any(|conn| conn.in_node().eq(in_node)))
        ).unwrap();

        let new_conn = conn::ConnGene::new(in_node.clone(), out_node.clone(), rand::random(), config.innov(in_node.clone(), out_node.clone()));
        (new_conn.in_node() as Self::NodeGene).forward_mut(|forward| forward.insert(new_conn.clone()));
        new_conn.out_node().backward_mut(|backward| backward.insert(new_conn.clone()));
        self.conn_genes.insert(new_conn.clone());

        new_conn.clone()
    }

    fn add_node_mut(&mut self, config: &Self::Config) -> Self::NodeGene {
        assert_ne!(self.conn_genes.len(), 0);

        let old_conn = self.conn_genes.iter().filter(|conn| conn.enabled()).choose(&mut rand::thread_rng()).unwrap();
        old_conn.set_enabled(false);

        let new_node = node::NodeGene::new_hidden();

        let conn_a = conn::ConnGene::new(old_conn.in_node(), new_node.clone(), 1.0, config.innov(old_conn.in_node(), new_node.clone()));
        let conn_b = conn::ConnGene::new(new_node.clone(), old_conn.out_node(), old_conn.weight(), config.innov(new_node.clone(), old_conn.out_node()));

        old_conn.in_node().forward_mut(|forward| forward.insert(conn_a.clone()));
        old_conn.out_node().backward_mut(|backward| backward.insert(conn_b.clone()));

        new_node.forward_mut(|forward| forward.insert(conn_b.clone()));
        new_node.backward_mut(|backward| backward.insert(conn_a.clone()));

        self.conn_genes.insert(conn_a.clone());
        self.conn_genes.insert(conn_b.clone());

        self.hidden_nodes.insert(new_node.clone());

        new_node.clone()
    }

    fn mut_conn_weight(&mut self, _config: &Self::Config) -> Self::ConnGene {
        todo!()
    }

    fn activate(&self, input: Vec<f32>, config: &Self::Config) -> Vec<f32> {
        assert_eq!(input.len(), config.input_len());

        let mut map = BTreeMap::from_iter(self.input_nodes.iter().enumerate().map(|(i, node)|
            (node.clone(), input.get(i).cloned().unwrap())
        ));

        while let Some((node, value)) = map.first_entry().map(|entry| (entry.key().clone(), entry.get().clone())) {
            let value = config.activation().activate(value);

            for conn in node.forward(|forward| forward.clone()).iter().filter(|conn| conn.enabled()) {
                map.entry(conn.out_node()).or_insert(0.0).add_assign(value * conn.weight());
            }

            map.remove(&node);
        }

        (0..config.output_len())
            .map(|i| config.activation().activate(map.get(self.output_nodes.get(i).unwrap()).cloned().unwrap()))
            .collect::<Vec<_>>().try_into().unwrap()
    }

    fn set_fitness(&mut self, fitness: f32, _: &Self::Config) {
        self.fitness = Some(fitness);
    }
    
    fn comp_dist(&self, other: &Self, config: &Self::Config) -> f32 {
        assert!(self.fitness.is_some());
        assert!(other.fitness.is_some());

        let n = cmp::max(self.conn_genes.first().cloned().unwrap().innov(), other.conn_genes.first().cloned().unwrap().innov());

        let (c1, c2, c3) = (config.c1(), config.c2(), config.c3());

        let e = self.excess_genes(&other).count() as f32;
        let d = self.disjoint_genes(&other).count() as f32;

        let mut w = self.matching_genes(&other).map(|(lhs, rhs)| lhs.weight().sub(&rhs.weight())).sum::<f32>();
        w /= self.matching_genes(&other).count() as f32;

        let mut dist = 0.0;
        dist += (c1 * e) / (n as f32);
        dist += (c2 * d) / (n as f32);
        dist += c3 * w;

        dist
    }
    
    fn crossover(&self, other: &Self, _config: &Self::Config) -> Self {
        // TODO: This works, but implementation is ugly. Clean it up in the future.

        assert!(self.fitness.is_some());
        assert!(other.fitness.is_some());

        let mut rng = rand::thread_rng();

        // The chance that the child genome inherits a matching connection gene from the more fit parent.
        const MATCHING_PREFERENCE: f64 = 2.0 / 3.0;

        let more_fit = cmp::max_by(self, other, |a, b| a.fitness.unwrap().total_cmp(&b.fitness.unwrap()));
        let less_fit = cmp::min_by(self, other, |a, b| a.fitness.unwrap().total_cmp(&b.fitness.unwrap()));

        match self.fitness.unwrap().total_cmp(&other.fitness.unwrap()) {
            cmp::Ordering::Less | cmp::Ordering::Greater => {
                for key in more_fit.conn_genes.intersection(&less_fit.conn_genes) {
                    let choice = match rng.gen_bool(MATCHING_PREFERENCE) {
                        true => more_fit.conn_genes.get(key).unwrap(),
                        false => less_fit.conn_genes.get(key).unwrap()
                    };

                    let conn = more_fit.conn_genes.get(key).unwrap();
                    conn.set_weight(choice.weight());
                    conn.set_enabled(choice.enabled());
                }

                Self {
                    conn_genes: more_fit.conn_genes.clone(),
                    input_nodes: more_fit.input_nodes.clone(),
                    hidden_nodes: more_fit.hidden_nodes.clone(),
                    output_nodes: more_fit.output_nodes.clone(),
                    fitness: None
                }
            },
            cmp::Ordering::Equal => {
                for key in self.conn_genes.intersection(&other.conn_genes) {
                    let choice = match rng.gen() {
                        true => self.conn_genes.get(key).unwrap(),
                        false => other.conn_genes.get(key).unwrap()
                    };

                    let conn = self.conn_genes.get(key).unwrap();
                    conn.set_weight(choice.weight());
                    conn.set_enabled(choice.enabled());
                }

                Self {
                    conn_genes: self.conn_genes.clone(),
                    input_nodes: self.input_nodes.clone(),
                    hidden_nodes: self.hidden_nodes.clone(),
                    output_nodes: self.output_nodes.clone(),
                    fitness: None
                }
            }
        }
    }
}