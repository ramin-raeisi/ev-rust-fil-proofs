use bellperson::{settings, groth16, bls::{Bls12}};
use std::time::{Instant};
use storage_proofs_core::{proof::ProofScheme};
use log::{info, trace};
use anyhow::{Result};
use std::fs;

pub fn calibrate_cpu_utilization<'a, S: ProofScheme<'a>>(pub_in: &S::PublicInputs, vanilla_proofs: Vec<S::Proof>, pub_params: &S::PublicParams,
     groth_params: &groth16::MappedParameters<Bls12>, function: &dyn Fn(&S::PublicInputs, Vec<S::Proof>,
         &S::PublicParams, &groth16::MappedParameters<Bls12>) -> Result<Vec<groth16::Proof<Bls12>>>)-> Result<()> 
    where
    S::Proof: Sync + Send,
    S::PublicParams: Sync + Send,
    S::PublicInputs: Clone + Sync,
    {
        golden_section_search::<'_, S>(0_f64, 1_f64, 0.1_f64, pub_in, vanilla_proofs.clone(), pub_params, groth_params, function);
        info!("golden search for cpu_utilization: finish");
        Ok(())
    }

fn golden_section_search<'a, S: ProofScheme<'a>>(f :f64, g:f64, tol:f64, pub_in: &S::PublicInputs, vanilla_proofs: Vec<S::Proof>, pub_params: &S::PublicParams,
    groth_params: &groth16::MappedParameters<Bls12>, function: &dyn Fn(&S::PublicInputs, Vec<S::Proof>,
        &S::PublicParams, &groth16::MappedParameters<Bls12>) -> Result<Vec<groth16::Proof<Bls12>>>)-> Result<()>
    where
    S::Proof: Sync + Send,
    S::PublicParams: Sync + Send,
    S::PublicInputs: Clone + Sync,
    {
    info!("golden search for cpu_utilization: start");
    let phi = ((5_f64).sqrt() - 1_f64) / 2_f64;
    let phi2 = (3_f64 - (5_f64).sqrt()) / 2_f64;
    let mut a = f;
    let mut b = g;
    let mut h = b - a;
    info!("phi  = {}, h = {}", phi, h);
    let n = ((tol / h).ln() / phi.ln()).ceil() as usize;
    info!{"n = {}",n};

    let mut c = a + phi2 * h;
    let mut d = a + phi * h;

    let mut filsettings = settings::Settings::new().unwrap();
    filsettings.cpu_utilization = c; 
    //let enc_filsettings = json::encode(&filsettings).unwrap();
    let mut enc_filsettings = serde_json::to_string(&filsettings)?;
    info!("CPU_UTILIZATION  = {}", c);
    fs::write("./fil-zk.config.toml", &enc_filsettings);

    info!("create config");

    let now = Instant::now();
    function(pub_in, vanilla_proofs.clone(), pub_params, groth_params);
    info!("circuit_proofs done");
    let mut yc = now.elapsed();
    enc_filsettings = serde_json::to_string(&filsettings)?;
    filsettings.cpu_utilization = d;
    info!("CPU_UTILIZATION  = {}", d); 
    fs::write("./fil-zk.config.toml", &enc_filsettings);
    let now = Instant::now();
    function(pub_in, vanilla_proofs.clone(), pub_params, groth_params);
    let mut yd = now.elapsed();

    for k in 0..n-1 {
        info!("k  = {}", k);
        if (yc < yd) {
            b = d;
            d = c;
            yd = yc;
            h = phi * h;
            c = a + phi2 * h;
            filsettings.cpu_utilization = c;
            info!("CPU_UTILIZATION  = {}", c);
            enc_filsettings = serde_json::to_string(&filsettings)?;  
            fs::write("./fil-zk.config.toml", &enc_filsettings);
            let now = Instant::now();
            function(pub_in, vanilla_proofs.clone(), pub_params, groth_params);
            yc = now.elapsed();
        }
        else {
            a = c;
            c = d;
            yc = yd;
            h = phi * h;
            d = a + phi * h;
            filsettings.cpu_utilization = d;
            info!("CPU_UTILIZATION  = {}", d);
            enc_filsettings = serde_json::to_string(&filsettings)?;  
            fs::write("./fil-zk.config.toml", &enc_filsettings);
            let now = Instant::now();
            function(pub_in, vanilla_proofs.clone(), pub_params, groth_params);
            yd = now.elapsed();
        }
    }
    if (yc < yd) {
        filsettings.cpu_utilization = a;  
        enc_filsettings = serde_json::to_string(&filsettings)?;
        fs::write("./fil-zk.config.toml", &enc_filsettings);
        info!("CPU_UTILIZATION  = {}", a);
    }
    else {
        filsettings.cpu_utilization = b;  
        enc_filsettings = serde_json::to_string(&filsettings)?;
        fs::write("./fil-zk.config.toml", &enc_filsettings);
        info!("CPU_UTILIZATION  = {}", b);
    }
    Ok(())
}