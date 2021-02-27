use bellperson::{settings, groth16, bls::{Bls12}};
use std::time::{Instant};
use storage_proofs_core::{proof::ProofScheme};
use log::{info};
use anyhow::{Result};
use std::fs;
use std::cmp::max;

pub fn calibrate_filsettings<'a, S: ProofScheme<'a>>(pub_in: &S::PublicInputs, vanilla_proofs: Vec<S::Proof>, pub_params: &S::PublicParams,
     groth_params: &groth16::MappedParameters<Bls12>, function: &dyn Fn(&S::PublicInputs, Vec<S::Proof>,
         &S::PublicParams, &groth16::MappedParameters<Bls12>) -> Result<Vec<groth16::Proof<Bls12>>>)-> Result<()> 
    where
    S::Proof: Sync + Send,
    S::PublicParams: Sync + Send,
    S::PublicInputs: Clone + Sync,
    {
        function(pub_in, vanilla_proofs.clone(), pub_params, groth_params);
        let f = vec![0_f64, 8_f64, 1_f64, 1_f64];
        let g = vec![1_f64, 14_f64, 3_f64, 4_f64];
        let tol = vec![0.1_f64, 1_f64, 1_f64, 1_f64];
        let mut filsettings = settings::Settings::new().unwrap();
        filsettings.size = 1;
        golden_section_search::<'_, S>(&f, &g, &tol, 2, &mut filsettings, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function);
        Ok(())
    }

fn golden_section_search<'a, S: ProofScheme<'a>>(f : &Vec<f64>, g: &Vec<f64>, tol_vec: &Vec<f64>, index: usize, filsettings: &mut settings::Settings, pub_in: &S::PublicInputs, vanilla_proofs: Vec<S::Proof>, pub_params: &S::PublicParams,
    groth_params: &groth16::MappedParameters<Bls12>, function: &dyn Fn(&S::PublicInputs, Vec<S::Proof>,
        &S::PublicParams, &groth16::MappedParameters<Bls12>) -> Result<Vec<groth16::Proof<Bls12>>>)-> Result<()>
    where
    S::Proof: Sync + Send,
    S::PublicParams: Sync + Send,
    S::PublicInputs: Clone + Sync,
    {
    let phi = ((5_f64).sqrt() - 1_f64) / 2_f64;
    let phi2 = (3_f64 - (5_f64).sqrt()) / 2_f64;
    let mut a = f[index];
    let mut b = g[index];
    let tol = tol_vec[index];
    let mut h = b - a;

    let mut c = (((a + phi2 * h) / tol).round())*tol;
    let mut d = (((a + phi * h) / tol).round())*tol;
    
    filsettings.set_value(index, a); 
    let mut enc_filsettings = serde_json::to_string(&filsettings)?;
    info!("parameter  = {}", a);
    fs::write("./fil-zk.config.toml", &enc_filsettings);
    if filsettings.size != 1 {
        filsettings.size -= 1;
        golden_section_search::<'_, S>(&f, &g, &tol_vec, index + 1, filsettings, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function);
        filsettings.size += 1;
    } 
    let now = Instant::now();
    function(pub_in, vanilla_proofs.clone(), pub_params, groth_params);
    let mut ya = now.elapsed();

    filsettings.set_value(index, b); 
    let mut enc_filsettings = serde_json::to_string(&filsettings)?;
    info!("parameter  = {}", b);
    fs::write("./fil-zk.config.toml", &enc_filsettings);
    if filsettings.size != 1 {
        filsettings.size -= 1;
        golden_section_search::<'_, S>(&f, &g, &tol_vec, index + 1, filsettings, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function);
        filsettings.size += 1;
    } 
    let now = Instant::now();
    function(pub_in, vanilla_proofs.clone(), pub_params, groth_params);
    let mut yb = now.elapsed();

    filsettings.set_value(index, c); 
    let mut enc_filsettings = serde_json::to_string(&filsettings)?;
    info!("parameter  = {}", c);
    fs::write("./fil-zk.config.toml", &enc_filsettings);
    if filsettings.size != 1 {
        filsettings.size -= 1;
        golden_section_search::<'_, S>(&f, &g, &tol_vec, index + 1, filsettings, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function);
        filsettings.size += 1;
    } 
    let now = Instant::now();
    function(pub_in, vanilla_proofs.clone(), pub_params, groth_params);
    let mut yc = now.elapsed();
    let mut yd;
   
    if d!=c {
        filsettings.set_value(index, d);
        enc_filsettings = serde_json::to_string(&filsettings)?;
        info!("parameter  = {}", d); 
        fs::write("./fil-zk.config.toml", &enc_filsettings);
        if filsettings.size != 1 {
            filsettings.size -= 1;
            golden_section_search::<'_, S>(&f, &g, &tol_vec, index + 1, filsettings, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function);
            filsettings.size += 1;
        } 
        let now = Instant::now();
        function(pub_in, vanilla_proofs.clone(), pub_params, groth_params);
        yd = now.elapsed();
    }
    else {
        yd = yc;
    }
    let mut k = -1;
    loop {
        k +=1;
        info!("k  = {}", k);
        if yc < yd {
            if d - a - tol <= tol * 0.1_f64 {
                if ya > yd {
                    filsettings.set_value(index, d);
                    info!("parameter  = {}", d);
                    enc_filsettings = serde_json::to_string(&filsettings)?;  
                    fs::write("./fil-zk.config.toml", &enc_filsettings);
                    break;
                }
                else {
                    filsettings.set_value(index, a);
                    info!("parameter  = {}", a);
                    enc_filsettings = serde_json::to_string(&filsettings)?;  
                    fs::write("./fil-zk.config.toml", &enc_filsettings);
                    break;
                }
            }
            b = d;
            yb = yd;
            d = c;
            yd = yc;
            h = b - a;
            c = (((a + phi2 * h)/tol).round()) * tol;
            if c == d {
                yc = yd;
            }
            else {
                filsettings.set_value(index, c);
                info!("parameter  = {}", c);
                enc_filsettings = serde_json::to_string(&filsettings)?;  
                fs::write("./fil-zk.config.toml", &enc_filsettings);
                if filsettings.size != 1 {
                    filsettings.size -= 1;
                    golden_section_search::<'_, S>(&f, &g, &tol_vec, index + 1, filsettings, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function);
                    filsettings.size += 1;
                } 
                let now = Instant::now();
                function(pub_in, vanilla_proofs.clone(), pub_params, groth_params);
                yc = now.elapsed();
            }
        }
        else if yc == yd {
                if yc < yb {
                    if yc < ya {
                        filsettings.set_value(index, c);
                        info!("parameter  = {}", c);
                        enc_filsettings = serde_json::to_string(&filsettings)?;  
                        fs::write("./fil-zk.config.toml", &enc_filsettings);
                        break;
                    }
                    else if ya > yb {
                        filsettings.set_value(index, b);
                        info!("parameter  = {}", b);
                        enc_filsettings = serde_json::to_string(&filsettings)?;  
                        fs::write("./fil-zk.config.toml", &enc_filsettings);
                        break;
                    }
                    else {
                        filsettings.set_value(index, a);
                        info!("parameter  = {}", a);
                        enc_filsettings = serde_json::to_string(&filsettings)?;  
                        fs::write("./fil-zk.config.toml", &enc_filsettings);
                    }
                }
        } 
        else {
            if b - c - tol <= tol * 0.1_f64 {
                if yb > yc {
                    filsettings.set_value(index, c);
                    info!("parameter  = {}", c);
                    enc_filsettings = serde_json::to_string(&filsettings)?;  
                    fs::write("./fil-zk.config.toml", &enc_filsettings);
                    break;
                }
                else {
                    filsettings.set_value(index, b);
                    info!("parameter  = {}", b);
                    enc_filsettings = serde_json::to_string(&filsettings)?;  
                    fs::write("./fil-zk.config.toml", &enc_filsettings);
                    break;
                }
            }
            a = c;
            ya = yc;
            c = d;
            yc = yd;
            h = phi * h;
            d = ((a + phi * h)/tol).round()*tol;
            if d == c{
                yd = yc;
            }
            else {
                filsettings.cpu_utilization = d;
                info!("parameter  = {}", d);
                enc_filsettings = serde_json::to_string(&filsettings)?;  
                fs::write("./fil-zk.config.toml", &enc_filsettings);
                if filsettings.size != 1 {
                    filsettings.size -= 1;
                    golden_section_search::<'_, S>(&f, &g, &tol_vec, index + 1, filsettings, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function);
                    filsettings.size += 1;
                } 
                let now = Instant::now();
                function(pub_in, vanilla_proofs.clone(), pub_params, groth_params);
                yd = now.elapsed();
            }
        }
    }
    Ok(())
}