use crypto_mac::Mac;
use hmac::Hmac;
use http::{HeaderMap, Method};
use sha1::Sha1;
use url::Url;
use time::now_utc;
use super::base64;
use log::{debug, info, trace, warn};

#[derive(Debug)]
pub struct Credential {
    access_key: String,
    secret_key: String,
}

impl Credential {
    pub fn new(ak: &str, sk: &str) -> Credential {
        Credential {
            access_key: ak.to_string(),
            secret_key: sk.to_string(),
        }
    }

    pub fn sign(&self, data: &[u8]) -> String {
        self.access_key.to_owned() + ":" + &self.base64_hmac_digest(data)
    }
    pub fn sign_s3_v2(&self, data: &[u8]) -> String {       
        format!("AWS {}:{}", self.access_key, self.base64_hmac_digest_s3(&data))
    }

    pub fn sign_with_data(&self, data: &[u8]) -> String {
        let encoded_data = base64::urlsafe(data);
        self.sign(encoded_data.as_bytes()) + ":" + &encoded_data
    }

    fn base64_hmac_digest(&self, data: &[u8]) -> String {
        let mut hmac = Hmac::<Sha1>::new_varkey(self.secret_key.as_bytes()).unwrap();
        hmac.input(data);
        base64::urlsafe(&hmac.result().code())
    }

    fn base64_hmac_digest_s3(&self, data: &[u8]) -> String {
        let mut hmac = Hmac::<Sha1>::new_varkey(self.secret_key.as_bytes()).unwrap();
        hmac.input(data);
        ::base64::encode_config(&hmac.result().code(), ::base64::STANDARD)
    }

    pub fn authorization_v1_for_request(
        &self,
        url_string: &str,
        content_type: &str,
        body: &[u8],
    ) -> Result<String, url::ParseError> {
        let authorization_token = self.sign_request_v1(url_string, content_type, body)?;
        Ok("QBox ".to_owned() + &authorization_token)
    }

    pub fn authorization_v2_for_request(
        &self,
        method: &Method,
        url_string: &str,
        headers: &HeaderMap,
        body: &[u8],
    ) -> Result<String, url::ParseError> {
        let authorization_token = self.sign_request_v2(method, url_string, headers, body)?;
        Ok("SDS ".to_owned() + &authorization_token)
    }

    pub fn sign_request_v1(
        &self,
        url_string: &str,
        content_type: &str,
        body: &[u8],
    ) -> Result<String, url::ParseError> {
        let u = Url::parse(url_string.as_ref())?;
        let mut data_to_sign = Vec::with_capacity(1024);
        data_to_sign.extend_from_slice(u.path().as_bytes());
        if let Some(query) = u.query() {
            if !query.is_empty() {
                data_to_sign.extend_from_slice(b"?");
                data_to_sign.extend_from_slice(query.as_bytes());
            }
        }
        data_to_sign.extend_from_slice(b"\n");
        if !content_type.is_empty() && !body.is_empty() {
            if Self::will_push_body_v1(content_type) {
                data_to_sign.extend_from_slice(body);
            }
        }
        Ok(self.sign(&data_to_sign))
    }

    pub fn sign_request_v2(
        &self,
        method: &Method,
        url_string: impl AsRef<str>,
        headers: &HeaderMap,
        body: &[u8],
    ) -> Result<String, url::ParseError> {
        let u = Url::parse(url_string.as_ref())?;
        let mut data_to_sign = Vec::with_capacity(1024);
        data_to_sign.extend_from_slice(method.as_str().as_bytes());
        data_to_sign.extend_from_slice(b" ");
        data_to_sign.extend_from_slice(u.path().as_bytes());
        if let Some(query) = u.query() {
            if !query.is_empty() {
                data_to_sign.extend_from_slice(b"?");
                data_to_sign.extend_from_slice(query.as_bytes());
            }
        }
        data_to_sign.extend_from_slice(b"\nHost: ");
        data_to_sign.extend_from_slice(
            u.host_str()
                .expect("Host must be existed in URL")
                .as_bytes(),
        );
        if let Some(port) = u.port() {
            data_to_sign.extend_from_slice(b":");
            data_to_sign.extend_from_slice(port.to_string().as_bytes());
        }
        data_to_sign.extend_from_slice(b"\n");

        if let Some(content_type) = headers.get("Content-Type") {
            data_to_sign.extend_from_slice(b"Content-Type: ");
            data_to_sign.extend_from_slice(content_type.as_ref());
            data_to_sign.extend_from_slice(b"\n");
            sign_data_for_x_sds_headers(&mut data_to_sign, headers);
            data_to_sign.extend_from_slice(b"\n");
            if !body.is_empty() && Self::will_push_body_v2(content_type.to_str().unwrap()) {
                data_to_sign.extend_from_slice(body);
            }
        } else {
            sign_data_for_x_sds_headers(&mut data_to_sign, &headers);
            data_to_sign.extend_from_slice(b"\n");
        }
        return Ok(self.sign(&data_to_sign));

        fn sign_data_for_x_sds_headers(data_to_sign: &mut Vec<u8>, headers: &HeaderMap) {
            let mut x_sds_headers = headers
                .iter()
                .map(|x| (x.0.as_str(), x.1.as_bytes()))
                .filter(|(key, _)| {key.len() > "x-sds-".len()})
                .filter(|(key, _)| key.starts_with("x-sds-"))
                .collect::<Vec<_>>();
            if x_sds_headers.is_empty() {
                return;
            }
            x_sds_headers.sort_unstable();
            for (header_key, header_value) in x_sds_headers {
                data_to_sign.extend_from_slice(header_key.as_ref());
                data_to_sign.extend_from_slice(b": ");
                data_to_sign.extend_from_slice(header_value);
                data_to_sign.extend_from_slice(b"\n");
            }
        }
    }

    pub fn sign_request_s3_v2(
        &self,
        method: &Method,
        resource: &str,
        headers: &HeaderMap,        
    ) -> (String,String) {     
        let mut data_to_sign = Vec::with_capacity(1024);
        //method
        data_to_sign.extend_from_slice(method.as_str().as_bytes());
        data_to_sign.extend_from_slice(b"\n\n");     
        //content type
        if let Some(content_type) = headers.get("Content-Type") {           
            data_to_sign.extend_from_slice(content_type.as_ref()); 
            
        } 
        data_to_sign.extend_from_slice(b"\n");  
        //date
        let date = now_utc().rfc822z().to_string();
        data_to_sign.extend_from_slice(date.as_bytes());   
        data_to_sign.extend_from_slice(b"\n"); 
        //resource      
        data_to_sign.extend_from_slice(resource.as_bytes());

        
        debug!("string to sign\n{}",String::from_utf8(data_to_sign.clone()).unwrap());

        return (self.sign_s3_v2(&data_to_sign),date);     
    }

    fn base64ed_hmac_digest(&self, data: &[u8]) -> String {
        let mut hmac = Hmac::<Sha1>::new_varkey(self.secret_key.as_bytes()).unwrap();
        hmac.input(data);
        base64::urlsafe(&hmac.result().code())
    }

    fn will_push_body_v1(content_type: &str) -> bool {
        super::FORM_MIME.eq_ignore_ascii_case(content_type)
    }

    fn will_push_body_v2(content_type: &str) -> bool {
        !super::BINARY_MIME.eq_ignore_ascii_case(content_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::header::HeaderValue;
    use std::{boxed::Box, error::Error, result::Result, sync::Arc, thread};

    #[test]
    fn test_sign() -> Result<(), Box<dyn Error>> {
        let credential = Arc::new(Credential::new("abcdefghklmnopq", "1234567890"));
        let mut threads = Vec::new();
        {
            threads.push(thread::spawn(move || {
                assert_eq!(
                    credential.sign(b"hello"),
                    "abcdefghklmnopq:b84KVc-LroDiz0ebUANfdzSRxa0="
                );
                assert_eq!(
                    credential.sign(b"world"),
                    "abcdefghklmnopq:VjgXt0P_nCxHuaTfiFz-UjDJ1AQ="
                );
            }));
        }
        {
            let credential = Arc::new(Credential::new("abcdefghklmnopq", "1234567890"));
            threads.push(thread::spawn(move || {
                assert_eq!(
                    credential.sign(b"-test"),
                    "abcdefghklmnopq:vYKRLUoXRlNHfpMEQeewG0zylaw="
                );
                assert_eq!(
                    credential.sign(b"ba#a-"),
                    "abcdefghklmnopq:2d_Yr6H1GdTKg3RvMtpHOhi047M="
                );
            }));
        }
        threads
            .into_iter()
            .for_each(|thread| thread.join().unwrap());
        Ok(())
    }

    #[test]
    fn test_sign_data() -> Result<(), Box<dyn Error>> {
        let credential = Arc::new(Credential::new("abcdefghklmnopq", "1234567890"));
        let mut threads = Vec::new();
        {
            let credential = credential.clone();
            threads.push(thread::spawn(move || {
                assert_eq!(
                    credential.sign_with_data(b"hello"),
                    "abcdefghklmnopq:BZYt5uVRy1RVt5ZTXbaIt2ROVMA=:aGVsbG8="
                );
                assert_eq!(
                    credential.sign_with_data(b"world"),
                    "abcdefghklmnopq:Wpe04qzPphiSZb1u6I0nFn6KpZg=:d29ybGQ="
                );
            }));
        }
        {
            let credential = credential.clone();
            threads.push(thread::spawn(move || {
                assert_eq!(
                    credential.sign_with_data(b"-test"),
                    "abcdefghklmnopq:HlxenSSP_6BbaYNzx1fyeyw8v1Y=:LXRlc3Q="
                );
                assert_eq!(
                    credential.sign_with_data(b"ba#a-"),
                    "abcdefghklmnopq:kwzeJrFziPDMO4jv3DKVLDyqud0=:YmEjYS0="
                );
            }));
        }
        threads
            .into_iter()
            .for_each(|thread| thread.join().unwrap());
        Ok(())
    }

    #[test]
    fn test_sign_request_s3_v2() -> Result<(), Box<dyn Error>> {
        let credential = Arc::new(Credential::new("s3a", "s3a"));
        let mut json_headers =  HeaderMap::new();
            
        let (signer,date)= credential.sign_request_s3_v2(
            &Method::GET,
            "/efe?wef",
            &json_headers
        );
        assert_eq!(
            signer,
            credential.sign_s3_v2(format!("GET\n\n\n{}\n/efe?wef",date).as_bytes()) 
        );
        

        json_headers.insert(
                "Content-Type",
                HeaderValue::from_static(super::super::JSON_MIME.into()),
            );
           
        let (signer,date)= credential.sign_request_s3_v2(
            &Method::GET,
            "/?wef",
            &json_headers
        );
        assert_eq!(
            signer,
            credential.sign_s3_v2(format!("GET\n\napplication/json\n{}\n/?wef",date).as_bytes()) 
        );
        //assert_eq!(credential.sign_s3_v2("HEAD\n\n\nMon, 23 Nov 2020 03:38:41 -0000\n/cdh-bk1/data-m-00001".as_bytes()),"AWS s3a:kCTuxslWuBbBEAHp3hkOIb1osz8=");
        Ok(())
    }
}
