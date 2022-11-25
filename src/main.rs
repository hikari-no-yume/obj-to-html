use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

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
    fn parse_from_str(s: &str) -> Result<Vector<N>, ()> {
        let mut res = [0f32; N];
        let mut split = s.split_whitespace();
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

impl Vector<3> {
    fn cross_product(self, rhs: Vector<3>) -> Vector<3> {
        Vector([
            self[1] * rhs[2] - self[2] * rhs[1],
            self[2] * rhs[0] - self[0] * rhs[2],
            self[0] * rhs[1] - self[1] * rhs[0],
        ])
    }
}

type Vertex = Vector<3>;

struct Face {
    vertices: [Vertex; 3],
    material: Option<usize>, // index into materials array
}

#[derive(Debug)]
struct Material {
    texture_path: Option<PathBuf>,
    factor: Vector<3>,
}

struct ObjParserState {
    vertices: Vec<Vertex>,
    vertex_range_min: Vector<3>,
    vertex_range_max: Vector<3>,
    mtl_path: Option<PathBuf>,
    materials: Vec<String>,
    faces: Vec<Face>,
}

fn parse_obj_data(state: &mut ObjParserState, data_type: &str, args: &str) {
    //println!("{}", data_type);

    match data_type {
        // material library (name of .mtl filename)
        "mtllib" => {
            state.mtl_path = Some(PathBuf::from(args));
        }
        // vertex, looks like "v 1.0 2.0 3.0"
        "v" => {
            let v = Vector::<3>::parse_from_str(args).unwrap();
            state.vertices.push(v);
            state.vertex_range_min = state.vertex_range_min.min(v);
            state.vertex_range_max = state.vertex_range_max.max(v);
        }
        // use named material from .mtl file
        "usemtl" => state.materials.push(String::from(args)),
        // face, looks like "f 1/1/1 2/2/2 3/3/3"
        "f" => {
            let mut vertices: [Vector<3>; 3] = Default::default();
            let mut vertex_split = args.split_whitespace();
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
            let face = Face {
                vertices,
                material: state.materials.len().checked_sub(1),
            };
            state.faces.push(face);
        }
        _ => (),
    }
}

struct MtlParserState {
    materials: HashMap<String, Material>,
    current_material: Option<String>,
}

fn parse_mtl_data(state: &mut MtlParserState, data_type: &str, args: &str) {
    match data_type {
        // new material
        "newmtl" => {
            state.current_material = Some(String::from(args));
            state.materials.insert(
                state.current_material.clone().unwrap(),
                Material {
                    texture_path: None,
                    factor: Vector::<3>([1f32, 1f32, 1f32]),
                },
            );
        }
        // diffuse colour factor, usually looks like "Kd 1 1 1"
        "Kd" => {
            assert!(!args.starts_with("spectral")); // unimplemented
            assert!(!args.starts_with("xyz")); // unimplemented
            let factor = Vector::<3>::parse_from_str(args).unwrap();
            state
                .materials
                .get_mut(state.current_material.as_ref().unwrap())
                .unwrap()
                .factor = factor;
        }
        // texture map for diffuse colour
        "map_Kd" => {
            state
                .materials
                .get_mut(state.current_material.as_ref().unwrap())
                .unwrap()
                .texture_path = Some(PathBuf::from(args));
        }
        _ => (),
    }
}

fn parse_file<P, F>(path: P, mut process_line: F)
where
    P: AsRef<std::path::Path>,
    F: FnMut(&str),
{
    let file = File::open(path).expect("Couldn't open file");
    let file = BufReader::new(file);

    // Using .split() rather than .lines() means UTF-8 decoding can be delayed
    // until after detecting comments, which should hopefully avoid problems
    // with files that have non-UTF-8 comments.
    for line in file.split(b'\n') {
        let line = line.expect("Couldn't read line from file");

        // comment line
        if let Some(b'#') = line.get(0) {
            continue;
        }

        let line = std::str::from_utf8(&line).expect("Non-comment lines should be UTF-8");
        let line = line.trim_end(); // might end with \r on Windows

        process_line(line);
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

    let mut obj_state = ObjParserState {
        vertices: Vec::new(),
        vertex_range_min: Vector::<3>([f32::INFINITY, f32::INFINITY, f32::INFINITY]),
        vertex_range_max: Vector::<3>([0f32, 0f32, 0f32]),
        mtl_path: None,
        materials: Vec::new(),
        faces: Vec::new(),
    };

    parse_file(&obj_path, |line| {
        let Some((data_type, args)) = line.split_once(' ') else {
            return;
        };
        let args = args.trim_start(); // might have extra whitespace

        parse_obj_data(&mut obj_state, data_type, args);
    });

    let mut mtl_state = MtlParserState {
        materials: HashMap::new(),
        current_material: None,
    };

    if let Some(relative_mtl_path) = obj_state.mtl_path {
        let mut mtl_path = PathBuf::from(obj_path);
        mtl_path.pop();
        mtl_path.push(relative_mtl_path);
        parse_file(mtl_path, |line| {
            let Some((data_type, args)) = line.split_once(' ') else {
                return;
            };
            let args = args.trim_start(); // might have extra whitespace

            parse_mtl_data(&mut mtl_state, data_type, args);
        });
    };

    let vertex_range = obj_state.vertex_range_max - obj_state.vertex_range_min;

    let target_dimension = 300.0; // rough pixel width of a cohost post?
    let scale = target_dimension / vertex_range[0].max(vertex_range[1]);

    let center = Vector::<3>([target_dimension / 2.0, target_dimension / 2.0, 0f32]);

    let offset = center - (vertex_range * scale) / 2.0 - obj_state.vertex_range_min * scale;
    // avoid having negative z values, it looks bad with perspective on
    let offset = Vector::<3>([offset[0], offset[1], vertex_range[2] * scale]);

    // spin animation from the cohost CSS (this does a Z-axis rotation)
    println!("<style>@keyframes spin {{to{{transform:rotate(360deg)}}}}</style>");

    println!("<div style=\"width: {}px; height: {}px; perspective: {}px; background: grey; position: relative; overflow: hidden;\">", target_dimension, target_dimension, vertex_range[2] * scale * 10.0);

    // continuously spin around the Y axis
    print!("<div style=\"transform-style: preserve-3d; transform: rotateX(-90deg);\">");
    print!("<div style=\"transform-style: preserve-3d; animation: 5s linear infinite spin;\">");
    println!("<div style=\"transform-style: preserve-3d; transform: rotateX(90deg);\">");

    eprintln!("{} triangles", obj_state.faces.len());
    for face in obj_state.faces {
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

        let Face {
            vertices: [a, b, c],
            material,
        } = face;

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

        // Technically we don't need to change the Z basis vector from
        // (0, 0, 1, 0) if we assume the input Z co-ordinate is always going to
        // be zero, but in practice we get exciting visual glitches in Chrome
        // and Firefox if we leave it like that. They're probably trying to
        // decompose the matrix and getting weird results.
        // So: set the Z basis vector to the normal.
        let z_basis_vector = x_basis_vector.cross_product(y_basis_vector);

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let matrix: [f32; 16] = [
            x_basis_vector[0], x_basis_vector[1], x_basis_vector[2], 0f32,
            y_basis_vector[0], y_basis_vector[1], y_basis_vector[2], 0f32,
            z_basis_vector[0], z_basis_vector[1], z_basis_vector[2], 0f32,
            translation[0], translation[1], translation[2], 1f32,
        ];

        let matrix = matrix.map(|f| format!("{:.5}", f)).join(",");

        let diffuse = if let Some(material_id) = material {
            let material_name = &obj_state.materials[material_id];
            mtl_state.materials[material_name].factor // TODO: sample texture
        } else {
            Vector::<3>([1f32, 1f32, 1f32])
        } * 255f32;

        println!("<div style=\"position: absolute; transform-origin: 0 0 0; transform: matrix3d({}); width: 0; height: 0; border-top: {:.5}px rgb({:.0}, {:.0}, {:.0}) solid; border-right: {:.5}px transparent solid;\"></div>", matrix, height, diffuse[0], diffuse[1], diffuse[2], width);
    }

    println!("</div></div></div>");

    println!("</div>");
}
