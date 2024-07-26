#![allow(dead_code)]

use bevy::prelude::*;
use bevy_neat::*;

const INPUT_LEN: usize = 2;
const OUTPUT_LEN: usize = 1;
const POP_SIZE: usize = 150;
const COMP_THRESH: f32 = 3.0;
const C1: f32 = 1.0;
const C2: f32 = 1.0;
const C3: f32 = 0.4;

const XOR_INPUTS: [(f32, f32); 4] = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
const XOR_OUTPUTS: [f32; 4] = [0.0, 1.0, 1.0, 0.0];

fn main() {
    App::new()
        .add_plugins(MinimalPlugins)
        .insert_resource(Population::<FeedForwardGenome>::new(DefaultConfig::new(
            INPUT_LEN, OUTPUT_LEN,
            POP_SIZE, COMP_THRESH,
            C1, C2, C3
        )))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut population: ResMut<Population<FeedForwardGenome>>) {
    population.run(|genome, config| {
        let mut fitness = 4.0;

        for (xor_input, xor_output) in XOR_INPUTS.into_iter().zip(XOR_OUTPUTS.into_iter()) {
            let output = genome.activate([xor_input.0, xor_input.1].into(), config).unwrap();
            fitness -= (output[0] - xor_output).powi(2);
        }

        fitness
    });
}
