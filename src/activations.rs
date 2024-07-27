use core::f32;
use std::ops::{Add, Div};
use crate::traits;

#[derive(Default)]
pub struct Sigmoid;

impl traits::Activation for Sigmoid {
    fn activate(self, x: f32) -> f32 {
        f32::consts::E.powf(x).div(1.0.add(f32::consts::E.powf(x)))
    }
}