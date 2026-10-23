use rustyneedle_needle::Needle;
use serde_json::Value;

fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q)),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w))),
        _ => a == b,
    }
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn environment_suites_match_the_recorded_engine() {
    let model = model::Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap();
    let goldens: Vec<Value> = testkit::golden("environments");
    let mut passed = 0;
    let mut total = 0;
    let mut report = Vec::new();
    let mut confidence_error = 0.0;
    for meta in goldens.iter().filter(|g| g.get("tool_schemas").is_some()) {
        let environment = meta["environment"].as_str().unwrap();
        let mut agent =
            Needle::builder().model(&model).tools(meta["tool_schemas"].as_array().unwrap().clone()).system(meta["system"].as_str().unwrap()).build();
        for case in goldens.iter().filter(|g| g["environment"] == environment && g.get("query").is_some()) {
            agent.reset();
            let got = agent.complete(case["query"].as_str().unwrap()).to_json();
            let want = &case["response"];
            let fields = ["function_calls", "suppressed_calls", "error_code"];
            let ok = fields.iter().all(|field| same(&got[*field], &want[*field])) && same(&got["validation"], &want["validation"]);
            confidence_error += (got["confidence"].as_f64().unwrap_or(0.0) - want["confidence"].as_f64().unwrap_or(0.0)).abs();
            total += 1;
            passed += ok as usize;
            if !ok {
                report.push(format!(
                    "{environment}: {}\n   want {} {} {} {}\n   got  {} {} {} {}",
                    case["query"],
                    want["function_calls"],
                    want["suppressed_calls"],
                    want["validation"],
                    want["confidence"],
                    got["function_calls"],
                    got["suppressed_calls"],
                    got["validation"],
                    got["confidence"]
                ));
            }
        }
    }
    println!("{}\n{passed}/{total} identical, mean confidence error {:.4}", report.join("\n"), confidence_error / total as f64);
    assert!(passed * 100 >= total * 85, "{passed}/{total}");
}
