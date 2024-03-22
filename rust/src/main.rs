use num::complex::Complex;
use tokio::task::JoinHandle;
use zfft::{fft, ifft, pvmul};

#[tokio::main]
async fn main() {

    let p : Vec<_> = (0..1024).map(|i| Complex::new(i,i)).collect();
    let q : Vec<_> = (0..1024).map(|i| Complex::new(-i,i)).collect();
    let r : Vec<_> = (0..1024).map(|i| Complex::new(i,-i)).collect();
    let s : Vec<_> = (0..1024).map(|i| Complex::new(-i,-i)).collect();

    let p_fft = tokio::spawn(fft(&p));
    let q_fft = tokio::spawn(fft(&q));
    let r_fft = tokio::spawn(fft(&r));
    let s_fft = tokio::spawn(fft(&s));

    let pq = tokio::spawn(pvmul(p_fft.await, q_fft.await));
    let rs = tokio::spawn(pvmul(r_fft.await, s_fft.await));

    let pqrs = tokio::spawn(pvmul(pq.await, rs.await));

    return ifft(pqrs);
}
