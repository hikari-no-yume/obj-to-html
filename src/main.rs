use image::io::Reader;
use image::{DynamicImage, GenericImageView, ImageOutputFormat, Rgb, RgbImage};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor};
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

    fn magnitude(self) -> f32 {
        let mut squares = 0f32;
        for i in 0..N {
            squares += self[i] * self[i];
        }
        squares.sqrt()
    }

    fn normalise(self) -> (f32, Vector<N>) {
        let magnitude = self.magnitude();

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

type Position = Vector<3>;
type UV = Vector<2>;
type Colour = Vector<3>;

#[derive(Default)]
struct Vertex {
    position: Position,
    uv: Option<UV>,
}

struct Face {
    vertices: [Vertex; 3],
    material: Option<usize>, // index into materials array
}

#[derive(Debug)]
struct Material {
    texture_path: Option<PathBuf>,
    factor: Colour,
}

struct ObjParserState {
    positions: Vec<Position>,
    position_range_min: Vector<3>,
    position_range_max: Vector<3>,
    uvs: Vec<UV>,
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
        // vertex position, looks like "v 1.0 2.0 3.0"
        "v" => {
            let v = Vector::<3>::parse_from_str(args).unwrap();
            state.positions.push(v);
            state.position_range_min = state.position_range_min.min(v);
            state.position_range_max = state.position_range_max.max(v);
        }
        // vertex texture UVs, looks like "vt 0.25 0.5"
        "vt" => {
            if let Ok(uv) = Vector::<2>::parse_from_str(args) {
                state.uvs.push(uv);
            } else if let Ok(uv) = Vector::<3>::parse_from_str(args) {
                assert!(uv[2] == 0f32);
                state.uvs.push(Vector([uv[0], uv[1]]));
            } else {
                panic!()
            }
        }
        // use named material from .mtl file
        "usemtl" => state.materials.push(String::from(args)),
        // face, looks like "f 1/1/1 2/2/2 3/3/3"
        "f" => {
            let mut vertices: [Vertex; 3] = Default::default();
            let mut vertex_split = args.split_whitespace();
            for i in 0..3 {
                let vertex_data = vertex_split
                    .next()
                    .expect("Should be three vertices in a face");
                let (position_id, uv_id_and_normal_id) = vertex_data.split_once('/').unwrap();
                let position_id: isize = position_id.parse().unwrap();
                let position_id: usize = (position_id - 1)
                    .try_into()
                    .expect("Negative indices unsupported");
                let position = *state
                    .positions
                    .get(position_id)
                    .expect("Position ID should be in range");

                let (uv_id, _normal_id) = uv_id_and_normal_id.split_once('/').unwrap();
                let uv = if uv_id.len() > 0 {
                    let uv_id: isize = uv_id.parse().unwrap();
                    let uv_id: usize = (uv_id - 1)
                        .try_into()
                        .expect("Negative indices unsupported");
                    Some(*state.uvs.get(uv_id).expect("UV ID should be in range"))
                } else {
                    None
                };

                vertices[i] = Vertex { position, uv };
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

fn get_texel(texture: &DynamicImage, coord: (i64, i64)) -> Colour {
    let x = coord.0.min((texture.width() - 1).into()).max(0) as u32;
    let y = coord.1.min((texture.height() - 1).into()).max(0) as u32;
    let pixel = texture.get_pixel(x, y);
    Vector([
        pixel[0] as f32 / 255f32,
        pixel[1] as f32 / 255f32,
        pixel[2] as f32 / 255f32,
    ])
}

fn lerp<const N: usize>(a: Vector<N>, b: Vector<N>, factor: f32) -> Vector<N> {
    a + (b - a) * factor
}

fn sample_texture(texture: &DynamicImage, uv: UV) -> Colour {
    let uv = Vector([
        if uv[0] >= 0.0 { uv[0] } else { 1.0 + uv[0] },
        if uv[1] >= 0.0 { uv[1] } else { 1.0 + uv[1] },
    ]);
    let uv = Vector([uv[0], 1.0 - uv[1]]);

    let uv = uv * Vector([texture.width() as f32, texture.height() as f32]);

    if false {
        // TODO: make this toggleable with a command-line flag
        let left = uv[0].floor() as i64;
        let right = uv[0].ceil() as i64;
        let x_factor = uv[0].fract();
        let top = uv[1].floor() as i64;
        let bottom = uv[1].ceil() as i64;
        let y_factor = uv[1].fract();

        let top_left = get_texel(texture, (left, top));
        let top_right = get_texel(texture, (right, top));
        let bottom_left = get_texel(texture, (left, bottom));
        let bottom_right = get_texel(texture, (right, bottom));

        // bilinear interpolation
        lerp(
            lerp(top_left, top_right, x_factor),
            lerp(bottom_left, bottom_right, x_factor),
            y_factor,
        )
    } else {
        // nearest neighbour
        get_texel(texture, (uv[0].round() as i64, uv[1].round() as i64))
    }
}

fn extract_triangle_texture(
    uvs: [UV; 3],
    triangle_size: Vector<2>,
    multiply_by: Colour,
    texture: &DynamicImage,
) -> String {
    let width = triangle_size[0];
    let height = triangle_size[1];
    assert!(width > 0.0);
    assert!(height > 0.0);
    let int_width = width.ceil() as u32;
    let int_height = height.ceil() as u32;

    let mut new_texture = RgbImage::new(int_width, int_height);

    let uv_origin = uvs[0];
    let uv_x_vector = uvs[1] - uvs[0];
    let uv_y_vector = uvs[2] - uvs[0];

    let top_left = Vector([0f32, 0f32]);
    let bottom_right = Vector([1f32, 1f32]);

    let mut last_sample = Vector([0f32, 0f32, 0f32]);
    for y in 0..int_height {
        for x in 0..int_width {
            let point = Vector([x as f32 / width, y as f32 / height]);

            let bottom_right_of_this_point =
                Vector([(x + 1) as f32 / width, (y + 1) as f32 / height]);
            let sample = if (bottom_right - bottom_right_of_this_point).magnitude()
                < (top_left - bottom_right_of_this_point).magnitude()
            {
                // This pixel is entirely outside the triangle, so it will be
                // hidden by the CSS clip path. To avoid wasting space in the
                // PNG, we can help the compressor out by filling the hidden
                // area with something similar to the neighbouring data but
                // internally uniform. The previous real pixel value works well.
                last_sample
            } else {
                let uv = uv_origin + point[0] * uv_x_vector + point[1] * uv_y_vector;
                let sample = sample_texture(texture, uv) * multiply_by;
                last_sample = sample;
                sample
            };
            let sample = sample * 255f32;
            new_texture.put_pixel(
                x,
                y,
                Rgb([sample[0] as u8, sample[1] as u8, sample[2] as u8]),
            );
        }
    }

    let mut png_buffer = Cursor::new(Vec::<u8>::new());
    new_texture.write_to(&mut png_buffer, ImageOutputFormat::Png);
    format!(
        "data:image/png;base64,{}",
        base64::encode(png_buffer.into_inner())
    )
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
        positions: Vec::new(),
        position_range_min: Vector::<3>([f32::INFINITY, f32::INFINITY, f32::INFINITY]),
        position_range_max: Vector::<3>([0f32, 0f32, 0f32]),
        uvs: Vec::new(),
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

    let mut textures: HashMap<PathBuf, DynamicImage> = HashMap::new();

    if let Some(relative_mtl_path) = obj_state.mtl_path {
        let mut mtl_path = PathBuf::from(obj_path);
        mtl_path.pop();
        mtl_path.push(relative_mtl_path);
        parse_file(&mtl_path, |line| {
            let line = line.trim_start();
            let Some((data_type, args)) = line.split_once(' ') else {
                return;
            };
            let args = args.trim_start(); // might have extra whitespace

            parse_mtl_data(&mut mtl_state, data_type, args);
        });

        for material in mtl_state.materials.values() {
            let Some(ref relative_texture_path) = material.texture_path else {
                continue;
            };
            if textures.contains_key(relative_texture_path) {
                continue;
            }

            let mut texture_path = PathBuf::from(&mtl_path);
            texture_path.pop();
            texture_path.push(relative_texture_path);

            let image = Reader::open(texture_path)
                .expect("Couldn't read texture file")
                .decode()
                .expect("Couldn't decode texture");
            eprintln!("{}, {}", image.width(), image.height());
            textures.insert(relative_texture_path.clone(), image);
        }
    };

    let position_range = obj_state.position_range_max - obj_state.position_range_min;

    let target_dimension = 300.0; // rough pixel width of a cohost post?
    let scale = target_dimension / position_range[0].max(position_range[1]);

    let center = Vector::<3>([target_dimension / 2.0, target_dimension / 2.0, 0f32]);

    let offset = center - (position_range * scale) / 2.0 - obj_state.position_range_min * scale;
    // avoid having negative z values, it looks bad with perspective on
    let offset = Vector::<3>([offset[0], offset[1], position_range[2] * scale]);

    // spin animation from the cohost CSS (this does a Z-axis rotation)
    println!("<style>@keyframes spin {{to{{transform:rotate(360deg)}}}}</style>");

    println!("<div style=\"width: {}px; height: {}px; perspective: {}px; background: grey; position: relative; overflow: hidden;\">", target_dimension, target_dimension, position_range[2] * scale * 10.0);

    // continuously spin around the Y axis
    print!("<div style=\"transform-style: preserve-3d; transform: translateZ({:.5}px) rotateX(-90deg);\">", -offset[2]);
    print!("<div style=\"transform-style: preserve-3d; animation: 5s linear infinite spin;\">");
    println!("<div style=\"transform-style: preserve-3d; transform: rotateX(90deg) translateZ({:5}px);\">", offset[2]);

    eprintln!("{} triangles", obj_state.faces.len());
    for face in obj_state.faces {
        // Let's call the points of the triangle face from the .obj file ABC.
        // The CSS trick gives us a unit right-angle triangle, let's call its
        // points DEF, where D is the top-left corner.
        // We want DEF * some transformation = ABC.
        // The vector DE is a unit vector pointing right, and the vector DF is
        // a unit vector pointing down. In other words, the basis vectors!
        // Conveniently, a transformation matrix consists of basis vectors.
        // If we apply the translation A, we can then use AB and AC as the
        // first two basis vectors in our transformation matrix, and in this way
        // transform DEF into ABC.

        let Face {
            vertices: [c, b, a], // reverse order to make facing test correct
            material,
        } = face;

        #[rustfmt::skip]
        let Vertex { position: a, uv: a_uv } = a;
        #[rustfmt::skip]
        let Vertex { position: b, uv: b_uv } = b;
        #[rustfmt::skip]
        let Vertex { position: c, uv: c_uv } = c;

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

        #[rustfmt::skip]
        let matrix: [f32; 16] = [
            x_basis_vector[0], x_basis_vector[1], x_basis_vector[2], 0f32,
            y_basis_vector[0], y_basis_vector[1], y_basis_vector[2], 0f32,
            z_basis_vector[0], z_basis_vector[1], z_basis_vector[2], 0f32,
            translation[0], translation[1], translation[2], 1f32,
        ];

        let matrix = matrix.map(|f| format!("{:.5}", f)).join(",");

        let (diffuse, texture) = if let Some(material_id) = material {
            let material_name = &obj_state.materials[material_id];
            let material = &mtl_state.materials[material_name];
            let diffuse = material.factor;
            if !a_uv.is_none() && !b_uv.is_none() && !c_uv.is_none() {
                let texture = material
                    .texture_path
                    .as_ref()
                    .and_then(|path| textures.get(path));
                (diffuse, texture)
            } else {
                (diffuse, None)
            }
        } else {
            (Vector::<3>([1f32, 1f32, 1f32]), None)
        };
        if let Some(texture) = texture {
            let url = extract_triangle_texture(
                [a_uv.unwrap(), b_uv.unwrap(), c_uv.unwrap()],
                Vector::<2>([width, height]),
                diffuse,
                texture,
            );

            println!("<div style=\"position: absolute; transform-origin: 0 0 0; transform: matrix3d({}); width: {:.5}px; height: {:.5}px; clip-path: polygon(0% 0%, 100% 0%, 0% 100%); background: url({}); backface-visibility: hidden;\"></div>", matrix, width, height, url);
        } else {
            let diffuse = diffuse * 255f32;

            println!("<div style=\"position: absolute; transform-origin: 0 0 0; transform: matrix3d({}); width: 0; height: 0; border-top: {:.5}px rgb({:.0}, {:.0}, {:.0}) solid; border-right: {:.5}px transparent solid; backface-visibility: hidden;\"></div>", matrix, height, diffuse[0], diffuse[1], diffuse[2], width);
        }
    }

    println!("</div></div></div>");

    println!("</div>");
}
