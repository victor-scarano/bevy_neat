use std::{cmp, collections::{BTreeMap, BTreeSet}, iter, ops::{Add, AddAssign, Div, Sub}};
use crate::*;
use bevy::ecs::component::Component;
use rand::seq::{IteratorRandom, SliceRandom};

pub trait Genome: Component + Sized {
    type Config: Config;
    type ConnGene;
    type NodeGene;

    fn new(config: Self::Config) -> Result<Self, ()>;

    fn mut_add_conn(&mut self, config: &Self::Config) -> Result<Self::ConnGene, ()>;

    fn mut_add_node(&mut self, config: &Self::Config) -> Result<Self::NodeGene, ()>;

    fn mut_conn_weight(&mut self, config: &Self::Config) -> Result<Self::ConnGene, ()>;

    fn activate(&self, input: Vec<f32>, config: &Self::Config) -> Result<Vec<f32>, ()>;

    fn set_fitness(&mut self, fitness: f32, config: &Self::Config) -> Result<(), ()>;

    fn crossover(&self, other: &Self, config: &Self::Config) -> Result<Self, ()>;

    fn comp_dist(&self, other: &Self, config: &Self::Config) -> Result<f32, ()>;
}

#[derive(Component, Debug)]
pub struct FeedForwardGenome {
    conn_genes: BTreeSet<ConnGene>,
    input_nodes: Vec<NodeGene>,
    hidden_nodes: BTreeSet<NodeGene>,
    output_nodes: Vec<NodeGene>,
    fitness: Option<f32>,
    innov: Innov,
}

impl FeedForwardGenome {
    fn matching_genes<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = (&ConnGene, &ConnGene)> + 'a {
        self.conn_genes.intersection(&other.conn_genes).map(|conn| (
            self.conn_genes.get(conn).unwrap(),
            other.conn_genes.get(conn).unwrap()
        ))
    }

    fn disjoint_genes<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &ConnGene> + 'a {
        let min_innov = cmp::min(
            self.conn_genes.first().unwrap().innov(),
            other.conn_genes.first().unwrap().innov()
        );

        self.conn_genes.symmetric_difference(&other.conn_genes)
            .filter(move |conn| conn.innov().le(&min_innov))
    }

    fn excess_genes<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &ConnGene> + 'a {
        let min_innov = cmp::min(
            self.conn_genes.first().unwrap().innov(),
            other.conn_genes.first().unwrap().innov()
        );

        self.conn_genes.symmetric_difference(&other.conn_genes).collect::<Vec<_>>()
            .into_iter().rev().take_while(move |conn| conn.innov().gt(&min_innov))
    }
}

impl Genome for FeedForwardGenome {
    type Config = DefaultConfig;
    type ConnGene = ConnGene;
    type NodeGene = NodeGene;

    fn new(config: Self::Config) -> Result<Self, ()> {
        Ok(Self {
            conn_genes: Default::default(),
            input_nodes: iter::repeat(NodeGene::new_input()).take(config.input_len()).collect(),
            hidden_nodes: Default::default(),
            output_nodes: iter::repeat(NodeGene::new_output()).take(config.output_len()).collect(),
            fitness: Default::default(),
            innov: Default::default(),
        })
    }

    fn mut_add_conn(&mut self, _config: &Self::Config) -> Result<Self::ConnGene, ()> {
        let mut possible_in_nodes = self.input_nodes.iter().chain(&self.hidden_nodes).cloned().collect::<Vec<_>>();
        if possible_in_nodes.len().eq(&0) { return Err(()); }
        possible_in_nodes.shuffle(&mut rand::thread_rng());

        let mut possible_out_nodes = self.output_nodes.iter().chain(&self.hidden_nodes).cloned().collect::<Vec<_>>();
        if possible_out_nodes.len().eq(&0) { return Err(()); }
        possible_out_nodes.shuffle(&mut rand::thread_rng());

        possible_in_nodes.iter().find(|node| possible_out_nodes.iter().count()
            .saturating_sub(self.hidden_nodes.contains(node) as usize)
            .saturating_sub(node.forward(|forward| forward.len()).unwrap())
            .gt(&0)
        ).and_then(|in_node| {
            let out_node = possible_out_nodes.iter()
                .find(|node| node.backward(|backward| !backward.iter().any(|c| c.in_node().eq(in_node))).unwrap())
                .unwrap();
            let new_conn = ConnGene::new(in_node.clone(), out_node.clone(), rand::random(), self.innov.next());
            new_conn.in_node().forward_mut(|forward| forward.insert(new_conn.clone())).unwrap();
            new_conn.out_node().backward_mut(|backward| backward.insert(new_conn.clone())).unwrap();
            self.conn_genes.insert(new_conn.clone());
            Some(new_conn.clone())
        }).ok_or(())
    }

    fn mut_add_node(&mut self, _config: &Self::Config) -> Result<Self::NodeGene, ()> {
        if self.conn_genes.len() == 0 { return Err(()); }

        let old_conn = self.conn_genes.iter().filter(|conn| conn.enabled()).choose(&mut rand::thread_rng()).unwrap();
        old_conn.disable();

        let new_node = NodeGene::new_hidden();

        let a = ConnGene::new(old_conn.in_node(), new_node.clone(), 1.0, self.innov.next());
        let b = ConnGene::new(new_node.clone(), old_conn.out_node(), old_conn.weight(), self.innov.next());

        old_conn.in_node().forward_mut(|forward| forward.insert(a.clone())).unwrap();
        old_conn.out_node().backward_mut(|backward| backward.insert(b.clone())).unwrap();

        self.conn_genes.insert(a.clone());
        self.conn_genes.insert(b.clone());

        new_node.forward_mut(|forward| forward.insert(b.clone())).unwrap();
        new_node.backward_mut(|backward| backward.insert(a.clone())).unwrap();

        self.hidden_nodes.insert(new_node.clone());

        Ok(new_node.clone())
    }

    fn mut_conn_weight(&mut self, _config: &Self::Config) -> Result<Self::ConnGene, ()> {
        todo!()
    }

    fn activate(&self, input: Vec<f32>, config: &Self::Config) -> Result<Vec<f32>, ()> {
        let mut map = BTreeMap::from_iter(
            self.input_nodes.iter().enumerate().map(|(i, node)| (node.clone(), input.get(i).cloned().unwrap()))
        );

        while let Some((node, value)) = map.first_key_value().map(|(node, value)| (node.clone(), value.clone())) {
            if node.is_output_node() { break; }

            for conn in node.forward(|forward| forward.clone()).unwrap().iter().filter(|conn| conn.enabled()) {
                map.entry(conn.out_node()).or_insert(0.0).add_assign(value * conn.weight());
                map.remove(&node);
            }
        }

        Ok((0..config.output_len())
            .map(|i| map.get(self.output_nodes.get(i).unwrap()).cloned().unwrap())
            .collect::<Vec<_>>().try_into().unwrap())
    }

    fn set_fitness(&mut self, fitness: f32, _: &Self::Config) -> Result<(), ()> {
        self.fitness = Some(fitness);
        Ok(())
    }

    fn crossover(&self, other: &Self, _config: &Self::Config) -> Result<Self, ()> {
        let matching = self.matching_genes(&other).map(|(lhs, rhs)| match rand::random() { true => lhs, false => rhs });
        
        let disjoint = self.disjoint_genes(&other)
            .filter_map(|conn| self.fitness.partial_cmp(&other.fitness).and_then(|ordering| match ordering {
                cmp::Ordering::Less => other.conn_genes.get(conn),
                cmp::Ordering::Equal => match rand::random() {
                    true => self.conn_genes.get(conn),
                    false => other.conn_genes.get(conn),
                },
                cmp::Ordering::Greater => self.conn_genes.get(conn),
            }));
        
        let excess = self.excess_genes(&other)
            .filter_map(|conn| self.fitness.partial_cmp(&other.fitness).and_then(|ordering| match ordering {
                cmp::Ordering::Less => other.conn_genes.get(conn),
                cmp::Ordering::Equal => match rand::random() {
                    true => self.conn_genes.get(conn),
                    false => other.conn_genes.get(conn),
                },
                cmp::Ordering::Greater => self.conn_genes.get(conn),
            }));
        
        // NOTE: The disabled genes may become enabled again in future generations: there's a preset chance that an
        // inherited gene is disabled if it is disabled in either parent.
        
        // We can ensure that both parent genomes will have the same number of input and output node genes because of
        // the input and output length generics.
        // We still need to sort out how the rest of the fields of the genome will be passed down to the child genome.
        // That includes input, hidden, and output node genes, as well as innovation.

        let _conn_genes = BTreeSet::from_iter(matching.chain(disjoint).chain(excess).cloned());
        // Ok(Self { conn_genes });
        Err(())
    }

    fn comp_dist(&self, other: &Self, _config: &Self::Config) -> Result<f32, ()> {
        if self.fitness.is_none() || other.fitness.is_none() {
            return Err(());
        }

        let mut n = cmp::max(
            self.conn_genes.first().cloned().unwrap().innov(),
            other.conn_genes.first().cloned().unwrap().innov()
        );

        if n.lt(&20) { n = 1; };

        let e = self.excess_genes(&other).count();
        let d = self.disjoint_genes(&other).count();

        let w = self.matching_genes(&other)
            .map(|(lhs, rhs)| lhs.weight().sub(&rhs.weight()))
            .sum::<f32>().div(self.matching_genes(&other).count() as f32);

        Ok((e as f32).div(n as f32).add((d as f32).div(n as f32)).add(w))
    }
}