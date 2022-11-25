use std::fs::File;
use std::io::{BufRead, BufReader};

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

#[derive(Copy, Clone, Debug)]
struct Vector<const N: usize>([f32; N]);

impl<const N: usize> Default for Vector<N> {
    fn default() -> Vector<N> {
        Vector([0f32; N])
    }
}

impl<const N: usize> std::ops::Index<usize> for Vector<N> {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}

macro_rules! impl_vec_op {
    ($trait:tt, $method:ident, $op:tt) => {
        impl<const N: usize> std::ops::$trait<Vector<N>> for Vector<N> {
            type Output = Vector<N>;
            fn $method(self, rhs: Vector<N>) -> Vector<N> {
                let mut res = [0f32; N];
                for i in 0..N {
                    res[i] = self[i] $op rhs[i];
                }
                Vector(res)
            }
        }
        impl<const N: usize> std::ops::$trait<f32> for Vector<N> {
            type Output = Vector<N>;
            fn $method(self, rhs: f32) -> Vector<N> {
                let mut res = [0f32; N];
                for i in 0..N {
                    res[i] = self[i] $op rhs;
                }
                Vector(res)
            }
        }
        impl<const N: usize> std::ops::$trait<Vector<N>> for f32 {
            type Output = Vector<N>;
            fn $method(self, rhs: Vector<N>) -> Vector<N> {
                let mut res = [0f32; N];
                for i in 0..N {
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

impl<const N: usize> Vector<N> {
    fn parse_from_str(s: &str, separator: char) -> Result<Vector<N>, ()> {
        let mut res = [0f32; N];
        let mut split = s.split(separator);
        for i in 0..N {
            let component = split.next().ok_or(())?;
            res[i] = component.parse().map_err(|_| ())?;
        }
        if !split.next().is_none() {
            Err(())?
        } else {
            Ok(Vector(res))
        }
    }

    fn min(self, rhs: Vector<N>) -> Vector<N> {
        let mut res = [0f32; N];
        for i in 0..N {
            res[i] = self[i].min(rhs[i]);
        }
        Vector(res)
    }
    fn max(self, rhs: Vector<N>) -> Vector<N> {
        let mut res = [0f32; N];
        for i in 0..N {
            res[i] = self[i].max(rhs[i]);
        }
        Vector(res)
    }

    fn normalise(self) -> (f32, Vector<N>) {
        let mut squares = 0f32;
        for i in 0..N {
            squares += self[i] * self[i];
        }
        let magnitude = squares.sqrt();

        if magnitude == 0f32 {
            (magnitude, self)
        } else {
            let mut res = self.0;
            for i in 0..N {
                res[i] /= magnitude;
            }
            (magnitude, Vector(res))
        }
    }
}

type Vertex = Vector<3>;

type Face = [Vertex; 3];

struct ParserState {
    vertices: Vec<Vertex>,
    vertex_range_min: Vector<3>,
    vertex_range_max: Vector<3>,
    faces: Vec<Face>,
}

fn parse_data(state: &mut ParserState, data_type: &str, args: &str) {
    //println!("{}", data_type);

    match data_type {
        // vertex, looks like "v 1.0 2.0 3.0"
        "v" => {
            let v = Vector::<3>::parse_from_str(args, ' ').unwrap();
            state.vertices.push(v);
            state.vertex_range_min = state.vertex_range_min.min(v);
            state.vertex_range_max = state.vertex_range_max.max(v);
        }
        // face, looks like "f 1/1/1 2/2/2 3/3/3"
        "f" => {
            let mut vertices: [Vector<3>; 3] = Default::default();
            let mut vertex_split = args.split(' ');
            for i in 0..3 {
                let vertex_data = vertex_split
                    .next()
                    .expect("Should be three vertices in a face");
                let (vertex_id, _normal_id_and_tex_coord_id) = vertex_data.split_once('/').unwrap();
                let vertex_id: isize = vertex_id.parse().unwrap();
                let vertex_id: usize = (vertex_id - 1)
                    .try_into()
                    .expect("Negative indices unsupported");
                vertices[i] = *state
                    .vertices
                    .get(vertex_id)
                    .expect("Vertex ID should be in range");
            }
            if !vertex_split.next().is_none() {
                panic!("Should be three vertices in a face")
            }
            state.faces.push(vertices);
        }
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
        faces: Vec::new(),
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

        parse_data(&mut state, data_type, args);
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

    for face in state.faces {
        // Let's call the points of the triangle face from the .obj file ABC.
        // The trick with CSS `border` gives us a unit right-angle triangle,
        // let's call its points DEF, where D is the top-left corner.
        // We want DEF * some transformation = ABC.
        // The vector DE is a unit vector pointing right, and the vector DF is
        // a unit vector pointing down. In other words, the basis vectors!
        // Conveniently, a transformation matrix consists of basis vectors.
        // If we apply the translation A, we can then use AB and AC as the
        // first two basis vectors in our transformation matrix, and in this way
        // transform DEF into ABC.

        let [a, b, c] = face;

        // Used to rotate everything so up is in the correct direction
        let flip_point = Vector::<3>([target_dimension, target_dimension, 0f32]);

        let a = flip_point - (a * scale + offset);
        let b = flip_point - (b * scale + offset);
        let c = flip_point - (c * scale + offset);

        let translation = a;
        let x_basis_vector = b - a;
        let y_basis_vector = c - a;

        // What was said above works in theory, but unfortunately, if we tell
        // tell the browser to make a 1px by 1px triangle and then scale it up
        // with transforms, we might actually get just a single 50% black pixel
        // quad, because the browser might pre-render the triangle to a 1px by
        // 1px texture and do the transform during composition.
        // So instead of using a unit triangle and scaling it with the matrix,
        // let's have a larger triangle and use unit vectors in the matrix.

        let (width, x_basis_vector) = x_basis_vector.normalise();
        let (height, y_basis_vector) = y_basis_vector.normalise();

        #[rustfmt_skip]
        let matrix: [f32; 16] = [
            x_basis_vector[0], x_basis_vector[1], x_basis_vector[2], 0f32,
            y_basis_vector[0], y_basis_vector[1], y_basis_vector[2], 0f32,
            0f32, 0f32, 1f32, 0f32,
            translation[0], translation[1], translation[2], 1f32,
        ];

        let matrix = matrix.map(|f| format!("{:.5}", f)).join(",");

        println!("<div style=\"position: absolute; transform-origin: 0 0 0; transform: matrix3d({}); width: 0; height: 0; border-top: {:.5}px black solid; border-right: {:.5}px transparent solid;\"></div>", matrix, height, width);
    }

    println!("</div></div></div>");

    println!("</div>");
}
