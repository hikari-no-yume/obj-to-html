use std::fs::File;
use std::io::{BufReader, BufRead};

trait ExpectNone {
    fn expect_none(&self, msg: &str);
}
impl<T> ExpectNone for Option<T> {
    fn expect_none(&self, msg: &str) {
        if self.is_some() {
            panic!("{}", msg);
        }
    }
}

#[derive(Copy, Clone)]
struct Vector<const T: usize>([f32; T]);

impl<const T: usize> std::ops::Index<usize> for Vector<T> {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}

macro_rules! impl_vec_op {
    ($trait:tt, $method:ident, $op:tt) => {
        impl<const T: usize> std::ops::$trait<Vector<T>> for Vector<T> {
            type Output = Vector<T>;
            fn $method(self, rhs: Vector<T>) -> Vector<T> {
                let mut res = [0f32; T];
                for i in 0..T {
                    res[i] = self[i] $op rhs[i];
                }
                Vector(res)
            }
        }
        impl<const T: usize> std::ops::$trait<f32> for Vector<T> {
            type Output = Vector<T>;
            fn $method(self, rhs: f32) -> Vector<T> {
                let mut res = [0f32; T];
                for i in 0..T {
                    res[i] = self[i] $op rhs;
                }
                Vector(res)
            }
        }
        impl<const T: usize> std::ops::$trait<Vector<T>> for f32 {
            type Output = Vector<T>;
            fn $method(self, rhs: Vector<T>) -> Vector<T> {
                let mut res = [0f32; T];
                for i in 0..T {
                    res[i] = self $op rhs[i];
                }
                Vector(res)
            }
        }
    }
}
impl_vec_op!(Add, add, +);
impl_vec_op!(Sub, sub, -);
impl_vec_op!(Mul, mul, *);
impl_vec_op!(Div, div, /);

impl<const T: usize> Vector<T> {
    fn min(self, rhs: Vector<T>) -> Vector<T> {
        let mut res = [0f32; T];
        for i in 0..T {
            res[i] = self[i].min(rhs[i]);
        }
        Vector(res)
    }
    fn max(self, rhs: Vector<T>) -> Vector<T> {
        let mut res = [0f32; T];
        for i in 0..T {
            res[i] = self[i].max(rhs[i]);
        }
        Vector(res)
    }
}

type Vertex = Vector<3>;

struct ParserState {
    vertices: Vec<Vertex>,
    vertex_range_min: Vector<3>,
    vertex_range_max: Vector<3>,
}

fn parse_triple(args: &[&str; 3]) -> Result<(f32, f32, f32), ()> {
    let x = args[0].parse().map_err(|_| ())?;
    let y = args[1].parse().map_err(|_| ())?;
    let z = args[2].parse().map_err(|_| ())?;
    Ok((x, y, z))
}

fn parse_data(state: &mut ParserState, data_type: &str, args: &[&str]) {
    //println!("{}", data_type);

    match data_type {
        "v" => {
            let (x, y, z) = parse_triple(args.try_into().unwrap()).unwrap();
            let v = Vector::<3>([x, y, z]);
            state.vertices.push(v);
            state.vertex_range_min = state.vertex_range_min.min(v);
            state.vertex_range_max = state.vertex_range_max.max(v);
        },
        _ => (),
    }
}

fn main() {
    let obj_path = {
        let mut args = std::env::args();
        let _ = args.next().unwrap(); // skip argv[0]
        let obj_path = args.next().expect("A .obj filename should be specified");
        args.next().expect_none("There should only be one argument");
        obj_path
    };

    let obj_file = File::open(obj_path).expect("Couldn't open file");
    let obj_file = BufReader::new(obj_file);

    let mut state = ParserState {
        vertices: Vec::new(),
        vertex_range_min: Vector::<3>([f32::INFINITY, f32::INFINITY, f32::INFINITY]),
        vertex_range_max: Vector::<3>([0f32, 0f32, 0f32]),
    };

    // Using .split() rather than .lines() means UTF-8 decoding can be delayed
    // until after detecting comments, which should hopefully avoid problems
    // with files that have non-UTF-8 comments.
    for line in obj_file.split(b'\n') {
        let line = line.expect("Couldn't read line from file");

        // comment line
        if let Some(b'#') = line.get(0) {
            continue;
        }

        let line = std::str::from_utf8(&line).expect("Non-comment lines should be UTF-8");
        let line = line.trim_end(); // might end with \r on Windows

        let Some((data_type, args)) = line.split_once(' ') else {
            continue;
        };

        let args: Vec<&str> = args.split(' ').collect();

        parse_data(&mut state, data_type, &args);
    }

    let vertex_range = state.vertex_range_max - state.vertex_range_min;


    let target_dimension = 300.0; // rough pixel width of a cohost post?
    let scale = target_dimension / vertex_range[0].max(vertex_range[1]);

    let center = Vector::<3>([target_dimension / 2.0, target_dimension / 2.0, 0f32]);

    let offset = center - (vertex_range * scale) / 2.0 - state.vertex_range_min * scale;
    // avoid having negative z values, it looks bad with perspective on
    let offset = Vector::<3>([offset[0], offset[1], vertex_range[2] * scale]);

    // spin animation from the cohost CSS (this does a Z-axis rotation)
    println!("<style>@keyframes spin {{to{{transform:rotate(360deg)}}}}</style>");

    println!("<div style=\"width: {}px; height: {}px; perspective: {}px; background: grey; position: relative; overflow: hidden;\">", target_dimension, target_dimension, vertex_range[2] * scale * 10.0);

    // continuously spin around the Y axis
    print!("<div style=\"transform-style: preserve-3d; transform: rotateX(-90deg);\">");
    print!("<div style=\"transform-style: preserve-3d; animation: 5s linear infinite spin;\">");
    println!("<div style=\"transform-style: preserve-3d; transform: rotateX(90deg);\">");

    for vertex in state.vertices {
        let vertex = vertex * scale + offset;
        // hack to fix Z and Y being wrong direction by rotating
        let vertex = Vector::<3>([target_dimension, target_dimension, 0f32]) - vertex;
        println!("<div style=\"position: absolute; translate: {}px {}px {}px; width: 4px; height: 4px; background: black;\"></div>", vertex[0], vertex[1], vertex[2]);
    }

    println!("</div></div></div>");

    println!("</div>");
}
