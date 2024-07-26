use std::iter;
use crate::*;
use bevy::ecs::system::Resource;

#[derive(Resource)]
pub struct Population<G: Genome> {
    config: G::Config,
    species: Vec<Species<G>>,
}

impl<G: Genome> Population<G> {
    pub fn new(config: G::Config) -> Self {
        Self {
            config: config.clone(),
            species: iter::once(Species {
                representative: G::new(config.clone()).unwrap(),
                shared_fitness: Default::default(),
                members: iter::repeat_with(|| G::new(config.clone()).unwrap()).take(config.pop_size()).collect()
            }).collect(),
        }
    }

    pub fn run(&mut self, fitness_fn: impl Fn(&G, &G::Config) -> f32) {
        for genome in self.species.iter_mut().flat_map(|species| species.members.iter_mut()) {
            genome.set_fitness(fitness_fn(&genome, &self.config), &self.config).unwrap();
        }
    }
}

pub struct Species<G: Genome> {
    representative: G,
    shared_fitness: Option<f32>,
    members: Vec<G>
}