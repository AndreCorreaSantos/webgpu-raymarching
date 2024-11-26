fn sdf_round_box(p: vec3<f32>, b: vec3<f32>, r: f32,quat: vec4f) -> f32 {
    let p_ = rotate_vector(p,quat);
    let q = abs(p_) - b + vec3<f32>(r, r, r);
    return length(max(q, vec3<f32>(0.0, 0.0, 0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

fn sdf_sphere(p: vec3f, r: vec4f, quat: vec4f) -> f32
{
  let p_ = rotate_vector(p,quat);
  return length(p_)-r.w;
}

fn sdf_torus(p: vec3f, t: vec2f, quat: vec4f) -> f32
{
  let p_ = rotate_vector(p,quat);
  let q = vec2f(length(p_.xz)-t.x,p_.y);
  return length(q)-t.y;
}

fn sdf_mandelbulb(p: vec3f) -> vec2f
{
  var w = p;
  var m = dot(w, w);

  var dz = 1.0;
  var i = 0;

  for (i = 0; i < 15; i = i + 1)
  {
    dz = 8.0 * pow(sqrt(m), 7.0) * dz + 1.0;
    var r = length(w);
    var b = 8.0 * acos(w.y / r);
    var a = 8.0 * atan2(w.x, w.z);
    w = p + pow(r, 8.0) * vec3f(sin(b) * sin(a), cos(b), sin(b) * cos(a));

    m = dot(w, w);
    if (m > 256.0)
    {
      break;
    }
  }
  var r = 0.25 * log(m) * sqrt(m) / dz;
  return vec2f(r, f32(i) / 16.0);
}

fn sdf_weird_thing(p_: vec3f, s: f32) -> f32
{
  var scale = 1.0;
  var orb = vec4f(1000.0);
  var p = p_;

  for (var i = 0; i < 8; i = i + 1)
  {
    p = -1.0 + 2.0 * modc(0.5 * p + 0.5, vec3f(1.0));

    var r2 = dot(p, p);
    orb = min(orb, vec4f(abs(p), r2));

    var k = s / r2;
    p *= k;
    scale *= k;
  }

  return 0.3 * abs(p.y) / scale;
}

fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_menger(p: vec3<f32>, time:f32) -> vec4<f32> {
    var slider = (2.0+sin(time*0.5))/2.0;
    var d = sdBox(p, vec3<f32>(slider*2.0)); 
    var color = vec3<f32>(0.1, 0.4, 1.0); // Initialize color (white by default)

    var s =1.0; // Scale factor
    for (var m = 0; m < 5; m++) { // Iterate 3 times
        let a = modc(p * s, vec3<f32>(2.0)) - vec3<f32>(1.0);

        s *= 3.0; // Increase scale

        let r = abs(1.0 - 3.0 * abs(a)); // Calculate distances to cross segments

        let da = max(r.x, r.y);
        let db = max(r.y, r.z);
        let dc = max(r.z, r.x);
        let c = (min(da, min(db, dc)) - 1.0) / s; // Compute fractal's distance function

        // Update the distance and color if closer
        if (c > d) {
            d = c;
            color = vec3<f32>(
                0.2 * da * db * dc, // Fake occlusion
                (1.0 + f32(m)) / 4.0, // Encodes iteration level in green
                1.0 // Static blue channel
            );
        }
    }

    return vec4<f32>(color, d); // Return color in xyz and distance in w
}
