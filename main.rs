use nannou::prelude::*;
use rand::prelude::*;

mod maze;
mod miccg;

fn main() {
    //let (a, b) = maze::maze(4, 4, 0.5);

    //println!("{:?}", a);

    //println!("{:?}", b);
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Fluid {
    let _window = app.new_window().view(view).build().unwrap();
    let size = app.main_window().rect().x_y_w_h();

    let mut fluid: Fluid = Fluid::new(
        4.7,
        size.2,
        size.3,
        size.0 - size.2 / 2.0,
        size.1 - size.3 / 2.0,
    );

    println!("{} Tiles in a {} by {} mesh", fluid.n, size.2, size.3);
    //fluid.vy[1000] = -1000.0;
    //fluid.smoke[1005] = 10000.0;
    //fluid.smoke[301] = 0.0;
    //fluid.smoke[202] = 10.0;
    //fluid.vy[3000] = 10.;

    /*for i in 0..fluid.cols * fluid.rows {
        let tile = &fluid.grid[i];
        println!(
            "Location:{}  {}  Fluid: {}  Pressure: {} Velocity: {} {}  Index: {}   ",
            tile.cx, tile.cy, tile.empty, tile.pressure, tile.vx, tile.vy, i
        )
    }*/
    fluid.build_maze(0., 0., 800., 800., 10, 10, 0.5, 10.);

    for i in 0..fluid.n {
        if (fluid.cx[i] + 150.0).powi(2) + (fluid.cy[i]).powi(2) < 70.0.powi(2) {
            // fluid.empty[i] = 1.0;
        } else if fluid.empty[i] == 1.0 {
            fluid.vx[i] = 3.0;
        }
        fluid.maxv = fluid.maxv.max(fluid.vx[i]).max(fluid.vy[i]);
    }
    fluid
}
// Circle: (fluid.cx[i] + 100.0).powi(2) + (fluid.cy[i]).powi(2) < 75.0.powi(2)
// Rectangle: (fluid.cx[i] + 100.0).abs() < 75.0 && (fluid.cy[i]).abs() < 75.0
/*|| (fluid.cx[i] - 200.0).abs() < 75.0 && (fluid.cy[i] + 400.).abs() < 300.0
|| (fluid.cx[i] - 200.0).abs() < 75.0 && (fluid.cy[i] - 400.).abs() < 300.0 */
fn update(_app: &App, model: &mut Fluid, _update: Update) {
    model.smoke[model.cols * 85 + 1] = 5.;
    //model.smoke[model.cols * 100 + 1] = 1.;

    for i in 1..model.rows - 1 {
        let index = i * model.cols;
        model.vx[index + 1] = 10.0;
        model.empty[index] = 1.0;
        //model.vx[index - 1] = 3.00;
        model.empty[index - 1] = 10.0;
        model.maxv = model.maxv.max(10.0);
    }
    let gravity = 1.1;
    let dt = model.timestep(gravity);
    //model.accelerate(0., 0., dt);
    model.vorticity_confinement(0.005, dt); //0.00168
    model.solve_incompressible();

    model.advect(dt);

    //println!("{}", model.maxv);

    //std::thread::sleep(std::time::Duration::new(0, 25000000));
}

fn view(app: &App, model: &Fluid, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);

    let mut norms = vec![0.0; model.n];
    let downscale = 1.;

    let mut minp = 100000000000.1;
    let mut maxp = -10000000000.1;
    let mut maxn = -100000.0;
    let mut minn = 100000.0;
    let mut minc = 1000000.0;
    let mut maxc = -100000000.0;
    for i in (model.cols..model.n - model.cols).step_by(downscale as usize) {
        if model.empty[i] == 1.0 {
            // Norm of velocity gathering
            let norm = (model.vx[i].powi(2) + model.vy[i].powi(2)).sqrt();
            norms[i] = norm;
            minn = minn.min(norm);
            maxn = maxn.max(norm);

            // max and min of pressure
            minp = minp.min(model.pressure[i]);
            maxp = maxp.max(model.pressure[i]);

            // max and min of curl
            let curl = model.curl(i);
            maxc = maxc.max(curl);
            minc = minc.min(curl);
        }
    }
    //println!("{}", max - min);
    for i in (model.cols..model.n - model.cols).step_by(downscale as usize) {
        if model.empty[i] == 1.0 {
            let pressure = ((model.pressure[i] - minp) / (maxp - minp)) as f32;
            let smoke = model.smoke[i] as f32;
            let norm = (((model.vx[i].powi(2) + model.vy[i].powi(2)).sqrt() - minn) / (maxn - minn))
                as f32;
            let curl = ((model.curl(i) - minc) / (maxc - minc)) as f32;
            draw.rect()
                .w_h(downscale * model.tilesize as f32, model.tilesize as f32)
                .x_y(model.cx[i] as f32, model.cy[i] as f32)
                .hsv(norm, 1.0, smoke);
        } else {
            draw.rect()
                .w_h(downscale * model.tilesize as f32, model.tilesize as f32)
                .x_y(model.cx[i] as f32, model.cy[i] as f32)
                .hsv(1.0, 0.0, 1.0);
        }
    }
    /*.hsv(0.8, 0.0, ((model.pressure[i] - min) / (max - min)) as f32*s/ */
    //model.smoke[i] as f32

    draw.to_frame(app, &frame).unwrap();
}

// holds a vector of tiles with length (mxn) including a padding boundary around the entire box
struct Fluid {
    rows: usize,
    cols: usize,
    n: usize,
    sizex: f64,
    sizey: f64,
    tilesize: f64,
    cx: Vec<f64>,
    cy: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    smoke: Vec<f64>,
    pressure: Vec<f64>,
    empty: Vec<f64>,
    maxv: f64,
}
impl Fluid {
    fn new(tilesize: f32, sizex: f32, sizey: f32, startx: f32, starty: f32) -> Fluid {
        let n = 2 + (sizey / tilesize).floor() as usize;
        let m = 2 + (sizex / tilesize).floor() as usize;
        let cx = (startx - tilesize / 2.0) as f64;
        let mut cy = (starty - tilesize / 2.0) as f64;
        let mut cx_vec = vec![0.0; n * m];
        let mut cy_vec = vec![0.0; n * m];
        let vx = vec![0.0; n * m];
        let vy = vec![0.0; n * m];
        let pressure = vec![0.0; n * m];
        let smoke = vec![0.0; n * m];
        let mut empty = vec![1.0; n * m];

        for j in 0..n {
            let mut cx_2 = cx;
            for i in 0..m {
                let index = j * m + i;
                if i == 0 || j == 0 || j == n - 1 || i == m - 1 {
                    empty[index] = 0.0;
                }

                cx_vec[index] = cx_2;
                cy_vec[index] = cy;

                cx_2 += tilesize as f64;
            }
            cy += tilesize as f64;
        }

        Fluid {
            rows: n,
            cols: m,
            n: n * m,
            sizex: sizex.into(),
            sizey: sizey.into(),
            tilesize: tilesize.into(),
            cx: cx_vec,
            cy: cy_vec,
            vx: vx,
            vy: vy,
            smoke: smoke,
            pressure: pressure,
            empty: empty,
            maxv: 0.0,
        }
    }
    // Modifies velocities within fluid so that it is incompressible
    fn solve_incompressible(&mut self) {
        self.maxv = 0.0;
        let gridsize = self.rows * self.cols;
        let arraysize = (self.rows - 2) * (self.cols - 2);
        let mut div = vec![0.0; arraysize];
        let mut precon = vec![0.0; gridsize]; // this is the diagonal matrix used for preconditioning in CG descent, maybe gridsize?

        let mut diag = vec![0.0; gridsize];
        let mut up_vec = vec![0.0; gridsize];
        let mut right_vec = vec![0.0; gridsize]; // helper lists that store if the left and down neighbors are solid or not

        //let mut trimatrix = TriMat::new((arraysize, arraysize));
        // loop over all tiles and calculate the net divergence for each

        let tuning = 0.97;
        for i in 1..self.rows - 1 {
            for j in 1..self.cols - 1 {
                let gridindex = i * self.cols + j;
                let arrayindex = (i - 1) * (self.cols - 2) + j - 1;
                if self.empty[gridindex] == 0.0 {
                    // not sure if this needs to be 1 or 0
                    //trimatrix.add_triplet(arrayindex, arrayindex, 1.0);
                    //diag[gridindex] = 1.;
                    continue;
                }
                let left = self.empty[gridindex - 1];
                let right = self.empty[gridindex + 1];
                let up = self.empty[gridindex + self.cols];
                let down = self.empty[gridindex - self.cols];

                let neighbors = up + down + left + right;

                // incomplete cholosky diagonal, will be modified incomplete later

                precon[gridindex] = neighbors;
                diag[gridindex] = neighbors;

                /*+ (left * precon[gridindex - 1]).powf(2.)
                + (down * precon[gridindex - self.cols]).powf(2.); // temporary, will be inverted later*/

                // Add velocity to the divergence if the neighboring tile is fluid
                div[arrayindex] = -(self.vx[gridindex + 1] * right - self.vx[gridindex] * left
                    + self.vy[gridindex + self.cols] * up
                    - self.vy[gridindex] * down);

                // maybe deal with pressure here also?

                if j < self.cols - 2 {
                    right_vec[gridindex] = -right;
                }

                if i < self.rows - 2 {
                    up_vec[gridindex] = -up;
                }
            }
        }
        //let divvector = ArrayBase::from_shape_vec((arraysize, 1), div);
        //println!("{:?}", divvector);

        //println!("{:?}", miccg::sparse_dot_dense(&sparse, &div));
        //println!("{}", sparse.to_dense());

        // finish the preconditioner
        // this is a terrible mess, but is derived from ubc's fluid simulation notes
        for i in 0..(self.rows - 2) {
            for j in 0..(self.cols - 2) {
                let gridindex = j + 1 + (i + 1) * self.cols;

                // incomplete cholosky condition

                precon[gridindex] -= (right_vec[gridindex - 1] * precon[gridindex - 1]).powi(2)
                    + (up_vec[gridindex - self.cols] * precon[gridindex - self.cols]).powi(2);

                // modified incomplete cholosky
                precon[gridindex] -= tuning
                    * (right_vec[gridindex - 1]
                        * up_vec[gridindex - 1]
                        * precon[gridindex - 1].powi(2)
                        + up_vec[gridindex - self.cols]
                            * right_vec[gridindex - self.cols]
                            * precon[gridindex - self.cols].powi(2));
                // inv square root to avoid division in application of preconditioner
                precon[gridindex] = 1.0 / (precon[gridindex] + f64::EPSILON).sqrt();
            }
        }
        // Solve the pressures for incompressibility
        div = miccg::miccg(
            &diag,
            &right_vec,
            &up_vec,
            &precon,
            &div,
            &self.pressure,
            self.rows,
            self.cols,
        );

        for i in 0..(self.rows - 2) {
            for j in 0..self.cols - 2 {
                let index = j + 1 + (i + 1) * (self.cols);
                self.pressure[index] = div[i * (self.cols - 2) + j]; // * self.empty[index]; // * might be needed, not sure
            }
        }

        // update velocities using solved pressures
        for i in 1..self.rows - 1 {
            for j in 1..self.cols - 1 {
                let index = i * self.cols + j;
                // velocity of solid wall is not affected by pressure

                self.vx[index] -= self.x_velocity_change(index, index - 1);

                self.vy[index] -= self.y_velocity_change(index, index - self.cols);

                self.maxv = self.maxv.max(self.vx[index]).max(self.vy[index]);
            }
        }

        // Add the new solved div values to the pressure?
    }
    // Updates all internal quantities of the fluid by using semi-lagrangian advection
    fn advect(&mut self, dt: f64) {
        let size = self.rows * self.cols;
        let mut vx_new = vec![0.0; size];
        let mut vy_new = vec![0.0; size];
        let mut smoke_new = vec![0.0; size];
        for i in 1..self.rows - 2 {
            for j in 1..self.cols - 2 {
                let gridindex = i * self.cols + j;
                if self.empty[gridindex] == 1.0 {
                    // x velocity
                    let vx = self.vx[gridindex];
                    let vy = (self.vy[gridindex]
                        + self.vy[gridindex - 1]
                        + self.vy[gridindex + self.cols]
                        + self.vy[gridindex + self.cols - 1])
                        * 0.25;
                    let x = (self.cx[gridindex] - self.tilesize / 2.0) - vx * dt;
                    let y = self.cy[gridindex] - vy * dt;

                    vx_new[gridindex] = self.interpolate(x, y, 0);

                    // y velocity
                    let vy = self.vy[gridindex];
                    let vx = (self.vx[gridindex]
                        + self.vx[gridindex + 1]
                        + self.vx[gridindex - self.cols]
                        + self.vx[gridindex - self.cols + 1])
                        * 0.25;
                    let x = self.cx[gridindex] - vx * dt;
                    let y = (self.cy[gridindex] - self.tilesize / 2.0) - vy * dt;

                    vy_new[gridindex] = self.interpolate(x, y, 1);

                    // smoke

                    let vx = (self.vx[gridindex] + self.vx[gridindex + 1]) / 2.;
                    let vy = (self.vy[gridindex] + self.vy[gridindex + self.cols]) / 2.;
                    let x = (self.cx[gridindex]) - vx * dt;
                    let y = (self.cy[gridindex]) - vy * dt;

                    smoke_new[gridindex] = self.interpolate(x, y, 2);
                }
            }
        }
        self.vx = vx_new;
        self.vy = vy_new;
        self.smoke = smoke_new;
    }
    // Bilinear interpolation of each quantity
    fn interpolate(&self, x: f64, y: f64, t: u8) -> f64 {
        let h1 = 1.0 / self.tilesize;
        let h2 = self.tilesize * 0.5;

        let x = (self.tilesize * (self.cols) as f64)
            .min(x + self.sizex * 0.5)
            .max(self.tilesize);
        let y = (self.tilesize * (self.rows) as f64).min((y + self.sizey * 0.5).max(self.tilesize));

        let (dx, dy, f) = match t {
            0 => (0.0, h2, &self.vx),
            1 => (h2, 0.0, &self.vy),
            2 => (h2, h2, &self.smoke),
            _ => panic!("Invalid Interpolation Field! 0:vx, 1:vy, 2:smoke"),
        };
        let x0 = (self.cols - 1).min(((x - dx + self.tilesize) * h1).floor() as usize);
        let tx = (x - dx + self.tilesize) * h1 - x0 as f64;
        let x1 = (self.cols - 1).min(x0 + 1);

        let y0 = (self.rows - 1).min(((y - dy + self.tilesize) * h1).floor() as usize);
        let ty = (y - dy + self.tilesize) * h1 - y0 as f64;
        let y1 = (self.rows - 1).min(y0 + 1);

        let sx = 1.0 - tx;
        let sy = 1.0 - ty;

        sx * sy * f[x0 + y0 * self.cols]
            + tx * sy * f[x1 + y0 * self.cols]
            + tx * ty * f[x1 + y1 * self.cols]
            + sx * ty * f[x0 + y1 * self.cols]
    }
    // Calculates a timestep based on the potential acceleration and max velocity in fluid
    fn timestep(&mut self, gravity: f64) -> f64 {
        return 5. * self.tilesize / (self.maxv + (5. * self.tilesize * gravity + 0.0001).sqrt());
    }
    // Updates x Velocities with respect to pressure as calculated for incompressibility
    fn x_velocity_change(&mut self, index1: usize, index2: usize) -> f64 {
        match (self.empty[index1] == 0.0, self.empty[index2] == 0.0) {
            (false, false) => self.pressure[index1] - self.pressure[index2],
            (true, _) => 0.0,
            (_, true) => self.vx[index1] - self.vx[index2],
        }
    }
    // Updates x Velocities with respect to pressure as calculated for incompressibility
    fn y_velocity_change(&mut self, index1: usize, index2: usize) -> f64 {
        match (self.empty[index1] == 0.0, self.empty[index2] == 0.0) {
            (false, false) => self.pressure[index1] - self.pressure[index2],
            (true, _) => 0.0,
            (_, true) => self.vy[index1] - self.vy[index2],
        }
    }

    // Calculates the curl about the center of a grid square
    fn curl(&self, index: usize) -> f64 {
        return self.vy[index + 1] - self.vy.get(index - 1).unwrap_or(&0.)
            + self.vx.get(index - self.cols).unwrap_or(&0.)
            - self.vx[index + self.cols];
    }

    // Calculates the vorticity in the x direciton around the center of a grid square
    fn vorticity_x(&self, vorticity: f64, index: usize) -> f64 {
        let curli = self.curl(index);
        let dy = self.curl(index + self.cols).abs() - (self.curl(index - self.cols)).abs();
        let dx = self.curl(index + 1).abs() - (self.curl(index - 1)).abs();
        let len = (dx.powi(2) + dy.powi(2)).sqrt() + 0.0000001;

        curli * self.tilesize * vorticity * dx / len
    }
    // Calculates the vorticity in the y direciton around the center of a grid square
    fn vorticity_y(&self, vorticity: f64, index: usize) -> f64 {
        let curli = self.curl(index);
        let dy = self.curl(index + self.cols).abs() - (self.curl(index - self.cols).abs());
        let dx = self.curl(index + 1).abs() - (self.curl(index - 1).abs());
        let len = (dx.powi(2) + dy.powi(2)).sqrt() + 0.0000001;

        curli * self.tilesize * vorticity * dy / len
    }
    // Adds a bit of velocity to areas with high vorticity, helps perserve small swirls and reduce numberical viscosity
    fn vorticity_confinement(&mut self, vorticity: f64, dt: f64) {
        for i in 3..self.rows - 4 {
            for j in 3..self.cols - 4 {
                let gridindex = i * self.cols + j;
                if self.empty[gridindex] == 1.0 {
                    self.vx[gridindex] += dt
                        * (self.vorticity_x(vorticity, gridindex)
                            + self.vorticity_x(vorticity, gridindex - 1))
                        * 0.5;
                    self.vy[gridindex] += dt
                        * (self.vorticity_y(vorticity, gridindex)
                            + self.vorticity_y(vorticity, gridindex - self.cols))
                        * 0.5;
                }
            }
            //println!("{}", gridindex);
        }
    }
    // Acclerates y velocities with respect to gravity
    fn accelerate(&mut self, gravityx: f64, gravityy: f64, dt: f64) {
        for i in 1..self.rows - 1 {
            for j in 1..self.cols - 1 {
                let gridindex = i * self.cols + j;
                if self.empty[gridindex] * self.empty[gridindex - self.cols] == 1.0 {
                    self.vy[gridindex] += gravityy * dt;
                    self.vx[gridindex] += gravityx * dt;
                }
            }
        }
    }

    fn build_maze(
        &mut self,
        centerx: f64,
        centery: f64,
        sizex: f64,
        sizey: f64,
        rows: usize,
        cols: usize,
        randweight: f64,
        wallsize: f64,
    ) {
        let mut walllist: Vec<(f64, f64, f64, f64)> = vec![];

        let (rwalls, bwalls) = maze::maze(rows, cols, randweight);
        let tilesizex = sizex / cols as f64;
        let tilesizey = sizey / rows as f64;

        let mut y = centery + sizey * 0.5;
        let mut x = centerx - sizex * 0.5;

        let starty = rand::thread_rng().gen_range(0..rows);
        let endy = rand::thread_rng().gen_range(0..rows);
        for i in 0..rows {
            if i != starty {
                walllist.push((x, y - tilesizey * 0.5, wallsize * 0.5, tilesizey * 0.5));
            }

            y -= tilesizey;
        }
        let mut y = centery + sizey * 0.5;
        for j in 0..cols {
            walllist.push((x + tilesizex * 0.5, y, tilesizex * 0.5, wallsize * 0.5));

            x += tilesizex;
        }
        for i in 0..rows {
            y -= tilesizey;
            x = centerx - sizex * 0.5;
            for j in 0..cols {
                x += tilesizex;
                if *rwalls[i].get(j).unwrap_or(&(i != endy)) {
                    walllist.push((x, y + tilesizey * 0.5, wallsize * 0.5, tilesizey * 0.5));
                }
                if bwalls.get(i).unwrap_or(&vec![true; cols])[j] {
                    walllist.push((x - tilesizex * 0.5, y, tilesizex * 0.5, wallsize * 0.5));
                }
            }
        }

        for i in 0..self.n {
            for wall in &walllist {
                if (self.cx[i] - wall.0).abs() < wall.2 && (self.cy[i] - wall.1).abs() < wall.3 {
                    self.empty[i] = 0.0;
                }
            }
        }
    }
}
