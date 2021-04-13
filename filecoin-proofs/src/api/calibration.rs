use bellperson::{settings, groth16, bls::{Bls12}};
use std::time::{Instant};
use storage_proofs_core::{proof::ProofScheme};
use log::{info};
use anyhow::{Result};
use std::fs;
use std::collections::HashMap;

pub fn calibrate_filsettings<'a, S: ProofScheme<'a>>(pub_in: &S::PublicInputs, vanilla_proofs: Vec<S::Proof>, pub_params: &S::PublicParams,
     groth_params: &groth16::MappedParameters<Bls12>, function: &dyn Fn(&S::PublicInputs, Vec<S::Proof>,
         &S::PublicParams, &groth16::MappedParameters<Bls12>) -> Result<Vec<groth16::Proof<Bls12>>>)-> Result<()> 
    where
    S::Proof: Sync + Send,
    S::PublicParams: Sync + Send,
    S::PublicInputs: Clone + Sync,
    {
        function(pub_in, vanilla_proofs.clone(), pub_params, groth_params)?;
        let f = vec![0_f64, 8_f64, 1_f64, 1_f64];
        let g = vec![1_f64, 14_f64, 3_f64, 4_f64];
        let tol = vec![0.1_f64, 1_f64, 1_f64, 1_f64];
        let mut indices = vec![1, 2, 3];
        let mut rec_depth = 3 as usize;
        settings::FILSETTINGS.lock().unwrap().cpu_utilization = 0_f64;
        golden_section_search::<S>(&f, &g, &tol, &indices, rec_depth, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function)?;
        indices = vec![0];
        rec_depth = 1 as usize;
        golden_section_search::<S>(&f, &g, &tol, &indices, rec_depth, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function)?;
        Ok(())
    }

fn golden_section_search<'a, S: ProofScheme<'a>>(f : &Vec<f64>, g: &Vec<f64>, tol_vec: &Vec<f64>, indices: &Vec<usize>, mut size: usize,  pub_in: &S::PublicInputs, vanilla_proofs: Vec<S::Proof>, pub_params: &S::PublicParams,
    groth_params: &groth16::MappedParameters<Bls12>, function: &dyn Fn(&S::PublicInputs, Vec<S::Proof>,
        &S::PublicParams, &groth16::MappedParameters<Bls12>) -> Result<Vec<groth16::Proof<Bls12>>>)-> Result<()>
    where
    S::Proof: Sync + Send,
    S::PublicParams: Sync + Send,
    S::PublicInputs: Clone + Sync,
    {
    let phi = ((5_f64).sqrt() - 1_f64) / 2_f64;
    let phi2 = (3_f64 - (5_f64).sqrt()) / 2_f64;
    let len = indices.len();
    let index = indices[len - size];
    let mut a = f[index];
    let mut b = g[index];
    let tol = tol_vec[index];
    let mut h = b - a;

    let mut c = (((a + phi2 * h) / tol).round())*tol;
    let mut d = (((a + phi * h) / tol).round())*tol;
    let mut timings= HashMap::new();

    settings::FILSETTINGS.lock().unwrap().set_value(index, c); 
    info!("parameter  = {}", c);
    if size != 1 {
        size -= 1;
        golden_section_search::<S>(&f, &g, &tol_vec, &indices, size, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function)?;
        size += 1;
    } 
    let now = Instant::now();
    function(pub_in, vanilla_proofs.clone(), pub_params, groth_params)?;
    let mut t = now.elapsed();
    timings.insert((c/tol) as usize, std::time::Duration::from(t));
   
    if d!=c {
        settings::FILSETTINGS.lock().unwrap().set_value(index, d);
        info!("parameter  = {}", d); 
        if size != 1 {
            size -= 1;
            golden_section_search::<S>(&f, &g, &tol_vec, &indices, size, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function)?;
            size += 1;
        } 
        let now = Instant::now();
        function(pub_in, vanilla_proofs.clone(), pub_params, groth_params)?;
        t = now.elapsed();
        timings.insert((d/tol) as usize, t);
    }
    let mut k = -1;
    loop {
        k +=1;
        info!("k  = {}", k);
        if timings.get(&((c/tol) as usize)) < timings.get(&((d/tol) as usize)) {
            if d - a - tol <= tol * 0.1_f64 {
                let at = timings.get(&((a/tol) as usize));
                match at {
                    Some(_) => {}
                    None => {
                        settings::FILSETTINGS.lock().unwrap().set_value(index, a);
                        info!("parameter  = {}", a);
                        if size != 1 {
                            size -= 1;
                            golden_section_search::<S>(&f, &g, &tol_vec, &indices, size, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function)?;
                        } 
                        let now = Instant::now();
                        function(pub_in, vanilla_proofs.clone(), pub_params, groth_params)?;
                        t = now.elapsed();
                        timings.insert((a/tol) as usize, t);
                    }
                }
                if timings.get(&((a/tol) as usize)) > timings.get(&((d/tol) as usize)) {
                    settings::FILSETTINGS.lock().unwrap().set_value(index, d);
                    info!("parameter  = {}", d);
                    let enc_filsettings = serde_json::to_string(&*settings::FILSETTINGS.lock().unwrap())?;  
                    fs::write("./fil-zk.config.toml", &enc_filsettings)?;
                    break;
                }
                else {
                    settings::FILSETTINGS.lock().unwrap().set_value(index, a);
                    info!("parameter  = {}", a);
                    let enc_filsettings = serde_json::to_string(&*settings::FILSETTINGS.lock().unwrap())?;  
                    fs::write("./fil-zk.config.toml", &enc_filsettings)?;
                    break;
                }
            }
            b = d;
            d = c;
            h = b - a;
            c = (((a + phi2 * h)/tol).round()) * tol;
            if c != d {
                settings::FILSETTINGS.lock().unwrap().set_value(index, c);
                info!("parameter  = {}", c);
                if size != 1 {
                    size -= 1;
                    golden_section_search::<S>(&f, &g, &tol_vec, &indices, size, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function)?;
                    size += 1;
                } 
                let now = Instant::now();
                function(pub_in, vanilla_proofs.clone(), pub_params, groth_params)?;
                t = now.elapsed();
                timings.insert((c/tol) as usize,t);
            }
        }
        else if timings.get(&((c/tol) as usize)) == timings.get(&((d/tol) as usize)) {
            let at = timings.get(&((a/tol) as usize));
            match at {
                Some(_) => {}
                None => {
                    settings::FILSETTINGS.lock().unwrap().set_value(index, a);
                    info!("parameter  = {}", a); 
                    if size != 1 {
                        size -= 1;
                        golden_section_search::<S>(&f, &g, &tol_vec, &indices, size, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function)?;
                        size += 1;
                    } 
                    let now = Instant::now();
                    function(pub_in, vanilla_proofs.clone(), pub_params, groth_params)?;
                    t = now.elapsed();
                    timings.insert((a/tol) as usize, t);
                }
            }
            let bt = timings.get(&((b/tol) as usize));
                match bt {
                    Some(_) => {}
                    None => {
                        settings::FILSETTINGS.lock().unwrap().set_value(index, b);
                        info!("parameter  = {}", b); 
                        if size != 1 {
                            size -= 1;
                            golden_section_search::<S>(&f, &g, &tol_vec, &indices, size, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function)?;
                        } 
                        let now = Instant::now();
                        function(pub_in, vanilla_proofs.clone(), pub_params, groth_params)?;
                        t = now.elapsed();
                        timings.insert((b/ tol) as usize, t);
                    }
                }
                if timings.get(&((c/tol) as usize)) < timings.get(&((b/tol) as usize))  {
                    if timings.get(&((c/tol) as usize)) < timings.get(&((a/tol) as usize)) {
                        settings::FILSETTINGS.lock().unwrap().set_value(index, c);
                        info!("parameter  = {}", c);
                        let enc_filsettings = serde_json::to_string(&*settings::FILSETTINGS.lock().unwrap())?;  
                        fs::write("./fil-zk.config.toml", &enc_filsettings)?;
                        break;
                    }
                    else {
                        settings::FILSETTINGS.lock().unwrap().set_value(index, a);
                        info!("parameter  = {}", a);
                        let enc_filsettings = serde_json::to_string(&*settings::FILSETTINGS.lock().unwrap())?;  
                        fs::write("./fil-zk.config.toml", &enc_filsettings)?;
                        break;
                    }
                }
                else if timings.get(&((b/tol) as usize)) < timings.get(&((a/tol) as usize)) {
                    settings::FILSETTINGS.lock().unwrap().set_value(index, b);
                    info!("parameter  = {}", b);
                    let enc_filsettings = serde_json::to_string(&*settings::FILSETTINGS.lock().unwrap())?;  
                    fs::write("./fil-zk.config.toml", &enc_filsettings)?;
                    break;
                }
                else {
                    settings::FILSETTINGS.lock().unwrap().set_value(index, a);
                    info!("parameter  = {}", a);
                    let enc_filsettings = serde_json::to_string(&*settings::FILSETTINGS.lock().unwrap())?;  
                    fs::write("./fil-zk.config.toml", &enc_filsettings)?;
                    break;
                }
        } 
        else {
            if b - c - tol <= tol * 0.1_f64 {
                let bt = timings.get(&((b/tol) as usize));
                match bt {
                    Some(_) => {}
                    None => {
                        settings::FILSETTINGS.lock().unwrap().set_value(index, b);
                        info!("parameter  = {}", b); 
                        if size != 1 {
                            size -= 1;
                            golden_section_search::<S>(&f, &g, &tol_vec, &indices, size, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function)?;
                        } 
                        let now = Instant::now();
                        function(pub_in, vanilla_proofs.clone(), pub_params, groth_params)?;
                        t = now.elapsed();
                        timings.insert((b/ tol) as usize, t);
                    }
                }
                if timings.get(&((b/tol) as usize)) > timings.get(&((c/tol) as usize)) {
                    settings::FILSETTINGS.lock().unwrap().set_value(index, c);
                    info!("parameter  = {}", c);
                    let enc_filsettings = serde_json::to_string(&*settings::FILSETTINGS.lock().unwrap())?;  
                    fs::write("./fil-zk.config.toml", &enc_filsettings)?;
                    break;
                }
                else {
                    settings::FILSETTINGS.lock().unwrap().set_value(index, b);
                    info!("parameter  = {}", b);
                    let enc_filsettings = serde_json::to_string(&*settings::FILSETTINGS.lock().unwrap())?;  
                    fs::write("./fil-zk.config.toml", &enc_filsettings)?;
                    break;
                }
            }
            a = c;
            c = d;
            h = b - a;
            d = ((a + phi * h)/tol).round()*tol;
            if d != c{
                settings::FILSETTINGS.lock().unwrap().cpu_utilization = d;
                info!("parameter  = {}", d);
                if size != 1 {
                    size -= 1;
                    golden_section_search::<S>(&f, &g, &tol_vec, &indices, size, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function)?;
                    size += 1;
                } 
                let now = Instant::now();
                function(pub_in, vanilla_proofs.clone(), pub_params, groth_params)?;
                t = now.elapsed();
                timings.insert((d/tol) as usize, t);
            }
        }
    }
    Ok(())
}