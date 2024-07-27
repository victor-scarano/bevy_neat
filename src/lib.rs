#![allow(dead_code)]

mod activations;
mod config;
mod conn;
mod genome;
mod node;
mod population;

pub use activations::*;
pub use config::*;
pub use conn::*;
pub use genome::*;
pub use node::*;
pub use population::*;

pub mod traits {
    use std::fmt::Debug;
    use bevy::ecs::component::Component;

    pub trait Genome: Clone + Component + Debug + PartialEq + Sized {
        type Config: Config<Self>;
        type ConnGene: ConnGene<Self>;
        type NodeGene: NodeGene;

        fn minimal(config: &Self::Config) -> Self;
        fn add_conn_mut(&mut self, config: &Self::Config) -> Self::ConnGene;
        fn add_node_mut(&mut self, config: &Self::Config) -> Self::NodeGene;
        fn mut_conn_weight(&mut self, config: &Self::Config) -> Self::ConnGene;
        fn activate(&self, input: Vec<f32>, config: &Self::Config) -> Vec<f32>;
        fn set_fitness(&mut self, fitness: f32, config: &Self::Config);
        fn crossover(&self, other: &Self, config: &Self::Config) -> Self;
        fn comp_dist(&self, other: &Self, config: &Self::Config) -> f32;
    }

    pub trait Config<G: Genome>: Clone + Sized {
        type Activation: Activation;

        fn innov(&self, in_node: G::NodeGene, out_node: G::NodeGene) -> u32;
        fn activation(&self) -> Self::Activation;
        fn input_len(&self) -> usize;
        fn output_len(&self) -> usize;
        fn pop_size(&self) -> usize;
        fn comp_thresh(&self) -> f32;
        fn c1(&self) -> f32;
        fn c2(&self) -> f32;
        fn c3(&self) -> f32;
    }

    pub trait ConnGene<G: Genome>: Debug + Eq + Send {
        fn in_node(&self) -> G::NodeGene;
        fn out_node(&self) -> G::NodeGene;
        fn weight(&self) -> f32;
        fn enabled(&self) -> bool;
        fn innov(&self) -> u32;
    }

    pub trait NodeGene: Clone + Debug + Send { }

    pub trait Activation: Default {
        fn activate(self, x: f32) -> f32;
    }
}