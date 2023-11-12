use rand::prelude::*;

#[derive(Clone, Debug)]
struct Cell {
    set_id: Option<usize>,
}

impl Cell {
    fn new() -> Cell {
        return Cell { set_id: None };
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Set {
    cells: Vec<usize>,
    goes_down: bool,
    id: usize,
}

impl Set {
    fn new(id: usize, col: usize) -> Set {
        Set {
            cells: vec![col],
            goes_down: false,
            id,
        }
    }
    fn join(&mut self, other: &Set, cells: &mut Vec<Cell>) {
        for col in &other.cells {
            self.cells.push(*col);
        }
        self.cells.sort();
        self.cells.dedup();
        for col in &self.cells {
            cells[*col].set_id = Some(self.id);
        }
    }

    fn new_inherited(&self, col: usize) -> Set {
        let mut new = self.clone();
        new.cells = vec![col];
        new.goes_down = false;
        return new;
    }
}

pub fn maze(rows: usize, cols: usize, randweight: f64) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
    let mut grid: Vec<Vec<Cell>> = vec![vec![Cell::new(); cols]; rows];

    let mut bottomwalls: Vec<Vec<bool>> = vec![];
    let mut rightwalls: Vec<Vec<bool>> = vec![];

    let mut setvec = vec![];
    let mut setcount = 0;

    for i in 0..grid[0].len() {
        let set = Set::new(setcount, i);
        setcount += 1;

        grid[0][i].set_id = Some(set.id);
        setvec.push(set);
    }

    for i in 0..rows - 1 {
        // assign empty cells to a new set
        for j in 0..cols {
            if grid[i][j].set_id == None {
                let set = Set::new(setcount, j);
                setcount += 1;

                grid[i][j].set_id = Some(set.id);
                setvec.push(set);
            }
        }
        // Join row randomly
        let mut rwall = vec![];
        for j in 0..cols - 1 {
            if grid[i][j + 1].set_id != grid[i][j].set_id {
                let wall = rand::random::<f64>() < randweight;

                rwall.push(wall);
                if !wall {
                    let mut startset = 100;
                    let mut endset = 1000;

                    for k in 0..setvec.len() {
                        if grid[i][j].set_id.unwrap() == setvec[k].id {
                            startset = k;
                        } else if grid[i][j + 1].set_id.unwrap() == setvec[k].id {
                            endset = k;
                        }
                    }
                    let temp = &setvec[endset].clone();
                    setvec[startset].join(temp, &mut grid[i]);
                    setvec.remove(endset);
                }
            } else {
                rwall.push(true);
                /*let mut startset = &mut Set::new(j, 0);
                let mut endset = &mut Set::new(j, 0);


                grid[i][j + 1].set_id = grid[i][j].set_id;
                for set in &mut setvec {
                    if grid[i][j].set_id.unwrap() == set.id {
                        startset = set;
                    } else if grid[i][j + 1].set_id.unwrap() == set.id {
                        endset = set;
                    }
                }
                startset.join(endset, &mut grid[i]);*/
            }
        }
        rightwalls.push(rwall);

        //ln!("{:?}", grid[i]);
        //println!("{:?}", setvec);
        // bottom walls
        let mut bwall = vec![];
        for j in 0..cols {
            let wall = rand::random::<f64>() < randweight;
            bwall.push(wall);

            for set in &mut setvec {
                if grid[i][j].set_id.unwrap() == set.id {
                    set.goes_down |= !wall;
                }
            }
        }
        for set in &mut setvec {
            if !set.goes_down {
                let index = set.cells.choose(&mut rand::thread_rng()).unwrap_or(&0);
                bwall[*index] = false;
            }
        }
        //println!("{:?}", setvec);
        // create new sets
        let mut newsets: Vec<Set> = vec![];
        for j in 0..cols {
            if !bwall[j] {
                let mut newset = &mut Set::new(0, 0);
                grid[i + 1][j].set_id = grid[i][j].set_id;
                for set in &mut setvec {
                    if grid[i][j].set_id.unwrap() == set.id {
                        newset = set;
                        break;
                    }
                }
                let newset = newset.new_inherited(j);
                let mut contains = false;
                for set in &mut newsets {
                    if newset.id == set.id {
                        set.join(&newset, &mut grid[i]);
                        contains = true;
                    }
                }
                if !contains {
                    newsets.push(newset);
                }
            }
        }
        //println!("{:?}", setvec);
        setvec = newsets;
        // New row
        bottomwalls.push(bwall);
    }
    //println!("{:?}", setvec);
    // assign empty cells to a new set
    for j in 0..cols {
        if grid[rows - 1][j].set_id == None {
            let set = Set::new(setcount, j);
            setcount += 1;

            grid[rows - 1][j].set_id = Some(set.id);
            setvec.push(set);
        }
    }

    // final row
    let mut rwall = vec![];
    for i in 0..cols - 1 {
        //println!("{:?}", grid[rows - 1]);
        let wall = grid[rows - 1][i].set_id == grid[rows - 1][i + 1].set_id;
        //println!("{}", wall);
        rwall.push(wall);
        if !wall {
            let mut startset = 100;
            let mut endset = 1000;

            for k in 0..setvec.len() {
                if grid[rows - 1][i].set_id.unwrap() == setvec[k].id {
                    startset = k;
                } else if grid[rows - 1][i + 1].set_id.unwrap() == setvec[k].id {
                    endset = k;
                }
            }
            //println!("{}", startset);
            let temp = &setvec[endset].clone();
            setvec[startset].join(temp, &mut grid[rows - 1]);
            setvec.remove(endset);
        }
    }
    rightwalls.push(rwall);
    //println!("{:?}", grid);

    (rightwalls, bottomwalls)
}
