use std::sync::{atomic::{AtomicU32, Ordering}, Arc};
use crate::{Activation, Sigmoid};

pub trait Config: Clone {
    type Innov;
    type Activation: Activation;

    fn innov(&self) -> Self::Innov;
    fn activation(&self) -> Self::Activation;
    fn input_len(&self) -> usize;
    fn output_len(&self) -> usize;
    fn pop_size(&self) -> usize;
    fn comp_thresh(&self) -> f32;
    fn c1(&self) -> f32;
    fn c2(&self) -> f32;
    fn c3(&self) -> f32;
}



#[derive(Clone)]
pub struct DefaultConfig {
    innov: Arc<Innov>,
    input_len: usize,
    output_len: usize,
    pop_size: usize,
    comp_thresh: f32,
    c1: f32,
    c2: f32,
    c3: f32,
}

impl DefaultConfig {
    pub fn new(
        input_len: usize, output_len: usize,
        pop_size: usize, comp_thresh: f32,
        c1: f32, c2: f32, c3: f32
    ) -> Self {
        Self {
            innov: Default::default(),
            input_len, output_len,
            pop_size, comp_thresh,
            c1, c2, c3
        }
    }
}

impl Config for DefaultConfig {
    type Innov = Arc<Innov>;
    type Activation = Sigmoid;

    fn innov(&self) -> Self::Innov { self.innov.clone() }
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