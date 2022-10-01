use f64ad_core::f64ad::*;

fn main() {
    let g = GlobalTape::get();
    let a = g.add_variable(2.0);
    // let b = g.add_variable(10.0);

    // a += b;
    // println!("{:?}", a);
    // println!("{:?}", a.node_idx);

    // let res = f64ad::f64ad_var(a.internal_sin().internal_sin());
    let res = f64ad::f64ad_var(a.internal_atanh());
    println!("{:?}", res);
    let grad = a.forward_mode_grad();
    let res = grad.wrt(&res);
    println!("{:?}", res);
    let grad = a.forward_mode_grad();
    let res = grad.wrt(&res);
    println!("{:?}", res);
    let grad = a.forward_mode_grad();
    let res = grad.wrt(&res);
    println!("{:?}", res);
    let grad = a.forward_mode_grad();
    let res = grad.wrt(&res);
    println!("{:?}", res);
    let grad = a.forward_mode_grad();
    let res = grad.wrt(&res);
    println!("{:?}", res);

    /*
    // let res = a * b;

    println!("{:?}", res);
    let grad = res.backwards_mode_grad();
    let res = grad.wrt(&a).unwrap();
    println!("{:?}", res);
    let grad = res.backwards_mode_grad();
    println!("{:?}", grad);
    let res = grad.wrt(&a).unwrap();
    println!("{:?}", res);
    let grad = res.backwards_mode_grad();
    let res = grad.wrt(&a).unwrap();
    println!("{:?}", res);
    // let grad = res.backwards_mode_grad();
    // let res = grad.wrt(&a).unwrap();
    // println!("{:?}", res);
    println!("{:?}", g.len());
    */
    /*
    println!("{:?}, {:?}", g.len(), res);
    let res = res.backwards_mode_grad();
    println!("{:?}, {:?}", g.len(), res);
    let r = res.wrt(&a).unwrap();
    let res = r.backwards_mode_grad();
    println!("{:?}, {:?}", g.len(), res);
    let r = res.wrt(&a).unwrap();
    let res = r.backwards_mode_grad();
    println!("{:?}, {:?}", g.len(), res);
    let r = res.wrt(&a).unwrap();
    let res = r.backwards_mode_grad();
    println!("{:?}, {:?}", g.len(), res);
    let r = res.wrt(&a).unwrap();
    let res = r.backwards_mode_grad();
    println!("{:?}, {:?}", g.len(), res);
    */
}