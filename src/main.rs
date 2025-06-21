use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rand::Rng;

fn main() {
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .expect("no output device available");

    let supported_configs_range = device
        .supported_output_configs()
        .expect("error while querying configs");

    let supported_config = supported_configs_range
        .filter(|config| config.sample_format() == cpal::SampleFormat::F32)
        .filter(|config| config.channels() == 1)
        .next()
        .expect("no supported config?!")
        .with_max_sample_rate();

    let config = supported_config.config();
    let mut oscillator = Noise::new(config.sample_rate.0);

    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                data.fill(0.0);
                oscillator.fill(data);
            },
            move |err| eprintln!("an error occurred on the output audio stream: {err}"),
            None,
        )
        .unwrap();

    stream.play().unwrap();

    _ = std::io::stdin().lines().next();
}

fn low_pass_filter(signal: &mut [rustfft::num_complex::Complex32], cutoff: f32, sample_rate: f32) {
    let n = signal.len();
    let freq_resolution = sample_rate / n as f32;

    for i in 0..n {
        let frequency = i as f32 * freq_resolution;
        if frequency > cutoff && frequency < (sample_rate - cutoff) {
            signal[i] *= 0.0;
        }
    }
}

fn high_pass_filter(signal: &mut [rustfft::num_complex::Complex32], cutoff: f32, sample_rate: f32) {
    let n = signal.len();
    let freq_resolution = sample_rate / n as f32;

    for i in 0..n {
        let frequency = i as f32 * freq_resolution;
        if frequency < cutoff || frequency > (sample_rate - cutoff) {
            signal[i] *= 0.0;
        }
    }
}

struct Sine {
    freq: f32,
    offset: usize,
}
impl Sine {
    fn new(freq: f32, sample_rate: f32) -> Sine {
        Sine {
            freq: freq * std::f32::consts::TAU / sample_rate,
            offset: 0,
        }
    }

    fn fill(&mut self, buffer: &mut [f32]) {
        buffer
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v += ((i + self.offset) as f32 * self.freq).sin());

        self.offset += buffer.len();
    }
}

struct Noise {
    offset: usize,
    buffer_size: Option<usize>,
    sample_rate: f32,
    buffer: Vec<rustfft::num_complex::Complex32>,
    scratch: Vec<rustfft::num_complex::Complex32>,
    fft: Option<std::sync::Arc<dyn rustfft::Fft<f32>>>,
    ifft: Option<std::sync::Arc<dyn rustfft::Fft<f32>>>,
}
impl Noise {
    fn new(sample_rate: u32) -> Noise {
        Noise {
            offset: 0,
            buffer_size: None,
            sample_rate: sample_rate as f32,
            buffer: Vec::new(),
            scratch: Vec::new(),
            fft: None,
            ifft: None,
        }
    }

    fn fill(&mut self, buffer: &mut [f32]) {
        if self.buffer_size.is_none() {
            self.buffer_size = Some(buffer.len());

            self.buffer.resize(
                self.buffer_size.unwrap(),
                rustfft::num_complex::Complex32::new(0.0, 0.0),
            );

            self.scratch.resize(
                self.buffer_size.unwrap(),
                rustfft::num_complex::Complex32::new(0.0, 0.0),
            );

            let mut planner: rustfft::FftPlanner<f32> = rustfft::FftPlanner::new();
            self.fft = Some(planner.plan_fft_forward(self.buffer_size.unwrap()));
            self.ifft = Some(planner.plan_fft_inverse(self.buffer_size.unwrap()));
        }

        let mut rng = rand::rng();

        self.buffer.iter_mut().for_each(|v| {
            *v = rustfft::num_complex::Complex32::new(rng.random_range(-1.0..1.0), 0.0)
        });

        // for normalizing the ffts
        let size = self.buffer_size.unwrap() as f32;

        self.fft
            .as_ref()
            .unwrap()
            .process_with_scratch(&mut self.buffer, &mut self.scratch);

        low_pass_filter(&mut self.buffer, 900.0, self.sample_rate);
        high_pass_filter(&mut self.buffer, 100.0, self.sample_rate);

        self.ifft
            .as_ref()
            .unwrap()
            .process_with_scratch(&mut self.buffer, &mut self.scratch);

        buffer
            .iter_mut()
            .zip(self.buffer.iter())
            .for_each(|(out, noise)| *out += noise.re / size);
        // .for_each(|(out, noise)| *out += noise.re);

        self.offset += buffer.len();
    }
}
