use std::{collections::BTreeSet, num::NonZeroUsize, sync::atomic::{AtomicU32, Ordering}};
use crate::{Sigmoid, traits::{self, ConnGene, Genome}};

#[derive(Clone)]
pub struct Config<G> where G: Genome + Clone, G::ConnGene: Ord {
    history: BTreeSet<G::ConnGene>,
    input_len: usize,
    output_len: usize,
    pop_size: usize,
    comp_thresh: f32, // Might want to change to NonZero?
    c1: f32,
    c2: f32,
    c3: f32,
}

impl<G> Config<G>
where
    G: Genome + Clone,
    G::ConnGene: Ord
{
    pub fn new(input_len: NonZeroUsize, output_len: NonZeroUsize, pop_size: NonZeroUsize, comp_thresh: f32, c1: f32, c2: f32, c3: f32) -> Self {
        Self {
            history: Default::default(),
            input_len: input_len.into(),
            output_len: output_len.into(),
            pop_size: pop_size.into(),
            comp_thresh, c1, c2, c3
        }
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
        self.history.iter()
            .find(|c| (c.in_node(), c.out_node()).eq(&(in_node.clone(), out_node.clone())))
            .and_then(|c| Some(c.innov()))
            .unwrap_or(self.history.first().cloned().and_then(|c| Some(c.innov())).unwrap_or(0))
    }

    fn activation(&self) -> Self::Activation { Default::default() }

    fn input_len(&self) -> usize { self.input_len.into() }

    fn output_len(&self) -> usize { self.output_len.into() }

    fn pop_size(&self) -> usize { self.pop_size.into() }

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