use std::{collections::BTreeSet, sync::atomic::{AtomicU32, Ordering}};
use crate::{Sigmoid, traits::{self, ConnGene, Genome}};

#[derive(Clone)]
pub struct Config<G> where G: Genome + Clone, G::ConnGene: Ord {
    conn_genes: BTreeSet<G::ConnGene>,
    input_len: usize,
    output_len: usize,
    pop_size: usize,
    comp_thresh: f32,
    c1: f32,
    c2: f32,
    c3: f32,
}

impl<G> Config<G>
where
    G: Genome + Clone,
    G::ConnGene: Ord
{
    pub fn new(input_len: usize, output_len: usize, pop_size: usize, comp_thresh: f32, c1: f32, c2: f32, c3: f32) -> Self {
        Self { conn_genes: Default::default(), input_len, output_len, pop_size, comp_thresh, c1, c2, c3 }
    }
}

impl<G> traits::Config<G> for Config<G>
where
    G: traits::Genome + Clone,
    G::ConnGene: Ord + Clone,
    G::NodeGene: PartialEq
{
    type Activation = Sigmoid;

    fn innov(&self, in_node: G::NodeGene, out_node: G::NodeGene) -> u32 {
        self.conn_genes.iter()
            .find(|c| (c.in_node(), c.out_node()).eq(&(in_node.clone(), out_node.clone())))
            .and_then(|c| Some(c.innov()))
            .unwrap_or(self.conn_genes.first().cloned().and_then(|c| Some(c.innov())).unwrap_or(0))
    }

    fn activation(&self) -> Self::Activation { Default::default() }

    fn input_len(&self) -> usize { self.input_len }

    fn output_len(&self) -> usize { self.output_len }

    fn pop_size(&self) -> usize { self.pop_size }

    fn comp_thresh(&self) -> f32 { self.comp_thresh }

    fn c1(&self) -> f32 { self.c1 }

    fn c2(&self) -> f32 { self.c2 }

    fn c3(&self) -> f32 { self.c3 }
}

#[derive(Debug, Default)]
pub struct Innov(AtomicU32);

impl Innov {
    pub fn next(&self) -> u32 { self.0.fetch_add(1, Ordering::Relaxed) }

    pub fn current(&self) -> u32 { self.0.load(Ordering::Relaxed) }
}