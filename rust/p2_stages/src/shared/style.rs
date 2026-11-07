use std::sync::OnceLock;

use babeldoc_il_types::{GraphicState, PassthroughValue, Shared};

pub struct PaletteColor {
    instruction: &'static str,
    value: OnceLock<Shared<GraphicState>>,
}

impl PaletteColor {
    pub const fn new(instruction: &'static str) -> Self {
        Self {
            instruction,
            value: OnceLock::new(),
        }
    }

    pub fn get(&self) -> Shared<GraphicState> {
        self.value
            .get_or_init(|| {
                Shared::new(GraphicState {
                    passthrough_per_char_instruction: Some(PassthroughValue::Materialized(
                        self.instruction.to_owned(),
                    )),
                    ..GraphicState::default()
                })
            })
            .clone()
    }
}

// Literal expectations copied from style_helper.py:30, 43, 47, 61, 71, 76, 86.
pub static WHITE: PaletteColor = PaletteColor::new("1 g 1 G");
pub static ORANGE: PaletteColor = PaletteColor::new(
    "1.0000000000 0.5843137255 0.0000000000 rg 1.0000000000 0.5843137255 0.0000000000 RG",
);
pub static YELLOW: PaletteColor = PaletteColor::new(
    "1.0000000000 0.8000000000 0.0000000000 rg 1.0000000000 0.8000000000 0.0000000000 RG",
);
pub static TEAL: PaletteColor = PaletteColor::new(
    "0.1882352941 0.6901960784 0.7803921569 rg 0.1882352941 0.6901960784 0.7803921569 RG",
);
pub static BLUE: PaletteColor = PaletteColor::new(
    "0.0000000000 0.4784313725 1.0000000000 rg 0.0000000000 0.4784313725 1.0000000000 RG",
);
pub static INDIGO: PaletteColor = PaletteColor::new(
    "0.3450980392 0.3372549020 0.8392156863 rg 0.3450980392 0.3372549020 0.8392156863 RG",
);
pub static PINK: PaletteColor = PaletteColor::new(
    "1.0000000000 0.1764705882 0.3333333333 rg 1.0000000000 0.1764705882 0.3333333333 RG",
);

#[cfg(test)]
mod style_tests;
