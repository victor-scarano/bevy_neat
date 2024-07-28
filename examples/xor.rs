// TODO: Insert plugin in main.

#![allow(dead_code)]

use std::ops::Add;
use bevy::prelude::*;
use bevy_neat::{FeedForwardGenome, Population, traits::Genome};

const XOR_INPUTS: [(f32, f32); 4] = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
const XOR_OUTPUTS: [f32; 4] = [0.0, 1.0, 1.0, 0.0];

fn main() {
    App::new()
        .add_plugins(MinimalPlugins)
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut population: ResMut<Population<FeedForwardGenome>>) {
    population.run(|genome, config| {
        XOR_INPUTS.into_iter().zip(XOR_OUTPUTS.into_iter()).map(|(xi, xo)| {
            let output = genome.activate([xi.0, xi.1].into(), config);
            -(output.first().cloned().unwrap() - xo).powi(2)
        }).sum::<f32>().add(4.0)
    });
}