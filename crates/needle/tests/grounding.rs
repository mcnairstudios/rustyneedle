mod common;

use chrono::TimeZone;
use common::{arguments, dated_call, envelope, finished, invoice_args, invoice_schema, order_schema, thermostat_schema, Stub};
use repair::Call;
use rustyneedle_needle::{extract_with_engine, source_years, with_date_fact, ExtractOptions, Needle, RunOptions};
use serde_json::{json, Value};

fn clock() -> chrono::DateTime<chrono::Local> {
    chrono::Local.with_ymd_and_hms(2026, 7, 21, 14, 30, 0).unwrap()
}

fn invoice_agent(engine: Box<dyn rustyneedle_needle::Engine>, system: &str) -> Needle {
    Needle::builder().engine(engine).tool(invoice_schema()).system(system).clock(clock()).build()
}

fn years(values: &[i32]) -> std::collections::BTreeSet<i32> {
    values.iter().copied().collect()
}

fn extract_invoice(
    envelopes: Vec<rustyneedle_needle::Response>,
    text: &str,
) -> Result<Option<serde_json::Map<String, Value>>, rustyneedle_needle::ExtractionValidationError> {
    let (engine, _) = Stub::new(envelopes);
    let options = ExtractOptions { system: Some("date: 2026-07-21 Tue 14:30".into()), ..ExtractOptions::default() };
    extract_with_engine(engine, invoice_schema(), text, &options)
}

fn total_envelope(total: f64, ungrounded: &[&str]) -> rustyneedle_needle::Response {
    envelope("Invoice", invoice_args(total), ungrounded)
}

#[test]
fn complete_flags_call_dates_that_contradict_the_input() {
    let (engine, _) = Stub::new(vec![dated_call("2026-09-05")]);
    let response = invoice_agent(engine, "").complete("Send an invoice to Acme due on 5th September 2031");
    assert_eq!(response.validation.unwrap().ungrounded, vec!["Invoice.due_date"]);
}

#[test]
fn complete_leaves_grounded_dates_alone() {
    let (engine, _) = Stub::new(vec![dated_call("2031-09-05")]);
    let response = invoice_agent(engine, "").complete("Send an invoice to Acme due on 5th September 2031");
    assert!(response.validation.is_none());
}

#[test]
fn input_without_a_year_is_not_checked() {
    let (engine, _) = Stub::new(vec![dated_call("2026-09-05")]);
    let response = invoice_agent(engine, "").complete("Send Acme an invoice due next Friday");
    assert!(response.validation.is_none());
}

#[test]
fn a_slash_date_does_not_mint_a_year_from_its_day_or_month() {
    let (engine, _) = Stub::new(vec![dated_call("2024-05-06")]);
    let response = invoice_agent(engine, "").complete("Send an invoice to Acme due on 5/6/24");
    assert!(response.validation.is_none());
    assert!(source_years("due on 5/6/24").is_empty());
    assert_eq!(source_years("due on 2024-05-06"), years(&[2024]));
}

#[test]
fn extract_accepts_a_date_written_with_slashes() {
    let result = extract_invoice(vec![dated_call("2024-12-31")], "Invoice from Acme Corp due 12/31/24").unwrap().unwrap();
    assert_eq!(result["due_date"], "2024-12-31");
}

#[test]
fn extract_licenses_a_relative_date_against_its_date_fact() {
    let text = "Invoice from Acme for the March 2019 order, due tomorrow";
    let (engine, _) = Stub::new(vec![dated_call("2026-07-22")]);
    let response = invoice_agent(engine, "").complete(text);
    assert!(response.validation.is_none());
    let result = extract_invoice(vec![dated_call("2026-07-22")], text).unwrap().unwrap();
    assert_eq!(result["due_date"], "2026-07-22");
}

#[test]
fn run_refuses_ungrounded_calls_unless_strict_is_off() {
    let query = "Send an invoice to Acme due on 5th September 2031";
    let (engine, handle) = Stub::new(vec![dated_call("2026-09-05"), finished()]);
    let mut agent = Needle::builder().engine(engine).tool_with_handler(invoice_schema(), |args| Ok(Value::Object(args.clone()))).clock(clock()).build();
    let response = agent.run(query, &RunOptions::default());
    assert_eq!(response.results.unwrap(), vec![json!({"error": "ungrounded due_date"})]);
    handle.load(vec![dated_call("2026-09-05"), finished()]);
    agent.reset();
    let loose = agent.run(query, &RunOptions { strict: false, ..RunOptions::default() });
    assert_eq!(loose.results.unwrap()[0]["due_date"], "2026-09-05");
}

#[test]
fn system_facts_license_relative_dates() {
    let (engine, _) = Stub::new(vec![dated_call("2026-07-22")]);
    let response = invoice_agent(engine, "date: 2026-07-21 Tue 14:30").complete("invoice Acme tomorrow for the 2019 reunion");
    assert!(response.validation.is_none());
}

#[test]
fn a_key_ending_in_date_does_not_suppress_the_date_fact() {
    let (engine, _) = Stub::new(vec![dated_call("2026-07-22")]);
    let agent = invoice_agent(engine, "user: Sam; last update: never");
    assert!(agent.system_text().starts_with("date: "));
    assert!(with_date_fact("candidate: Sam", clock()).starts_with("date: "));
    for system in ["date: 2026-07-21 Tue 14:30", "locale: en-US; date: Tue"] {
        assert_eq!(with_date_fact(system, clock()), system);
    }
}

#[test]
fn years_carry_across_turns_until_reset() {
    let (engine, _) = Stub::new(vec![dated_call("2031-09-05")]);
    let mut agent = invoice_agent(engine, "");
    assert!(agent.complete("bill Acme on 5th September 2031").validation.is_none());
    assert!(agent.complete(&json!({"id": 42, "since": "2019-03-01"}).to_string()).validation.is_none());
    agent.reset();
    let response = agent.complete("customer since March 2019, bill them");
    assert_eq!(response.validation.unwrap().ungrounded, vec!["Invoice.due_date"]);
}

#[test]
fn engine_reported_fabrications_are_kept_and_block_execution() {
    let mut flagged = dated_call("2031-09-05");
    flagged.validation = Some(rustyneedle_needle::Validation { ungrounded: vec!["Invoice.vendor".into()], negation: None });
    let (engine, handle) = Stub::new(vec![flagged, finished()]);
    let mut agent = Needle::builder().engine(engine).tool_with_handler(invoice_schema(), |args| Ok(Value::Object(args.clone()))).clock(clock()).build();
    let response = agent.run("bill someone on 5th September 2031", &RunOptions::default());
    assert_eq!(response.results.unwrap(), vec![json!({"error": "ungrounded vendor"})]);
    assert_eq!(handle.calls.lock().unwrap().last().unwrap(), "[{\"error\": \"ungrounded vendor\"}]");
}

#[test]
fn extract_clears_a_thousands_separated_number() {
    let result = extract_invoice(vec![total_envelope(1200.0, &["Invoice.total"])], "Invoice from Acme Corp, $1,200.00, due 2026-09-01").unwrap().unwrap();
    assert_eq!(result["total"], 1200.0);
    assert_eq!(result["due_date"], "2026-09-01");
}

#[test]
fn extract_clears_a_thousands_separated_integer() {
    let result = extract_invoice(
        vec![envelope("Invoice", json!({"vendor": "Acme Corp", "total": 1200, "due_date": "2026-09-01"}), &["Invoice.total"])],
        "Invoice from Acme Corp, $1,200, due 2026-09-01",
    );
    assert_eq!(result.unwrap().unwrap()["total"], 1200);
}

#[test]
fn extract_leaves_an_unseparated_number_working() {
    let result = extract_invoice(vec![total_envelope(1200.0, &["Invoice.total"])], "Invoice from Acme Corp, $1200.00, due 2026-09-01").unwrap().unwrap();
    assert_eq!(result["total"], 1200.0);
}

#[test]
fn extract_still_rejects_a_fabricated_number() {
    let error = extract_invoice(vec![total_envelope(9999.0, &["Invoice.total"])], "Invoice from Acme Corp, $1,200.00, due 2026-09-01").unwrap_err();
    assert!(error.to_string().contains("total"));
}

#[test]
fn extract_still_rejects_a_non_numeric_engine_flag() {
    let error = extract_invoice(vec![total_envelope(1200.0, &["Invoice.vendor"])], "Invoice from Acme Corp, $1,200.00, due 2026-09-01").unwrap_err();
    assert!(error.to_string().contains("vendor"));
}

#[test]
fn extract_clears_a_nested_separated_number() {
    let (engine, _) = Stub::new(vec![envelope("Order", json!({"items": [{"price": 1100.0}]}), &["Order.items[0].price"])]);
    let options = ExtractOptions { system: Some("date: 2026-07-21 Tue 14:30".into()), ..ExtractOptions::default() };
    let result = extract_with_engine(engine, order_schema(), "Order with one item priced $1,100.00", &options).unwrap().unwrap();
    assert_eq!(result["items"][0]["price"], 1100.0);
}

#[test]
fn extract_does_not_clear_a_digit_run_inside_a_longer_number() {
    for source in [
        "Invoice from Acme Corp, $12,000.00, due 2026-09-01",
        "Invoice from Acme Corp, $11,200.00, due 2026-09-01",
        "Invoice from Acme Corp, $1,2000.00, due 2026-09-01",
    ] {
        assert!(extract_invoice(vec![total_envelope(1200.0, &["Invoice.total"])], source).is_err(), "{source}");
    }
}

#[test]
fn extract_does_not_clear_a_number_absent_from_the_source() {
    assert!(extract_invoice(vec![total_envelope(1200.0, &["Invoice.total"])], "Invoice from Acme Corp, due 2026-09-01").is_err());
}

#[test]
fn extract_does_not_read_a_date_hyphen_as_a_minus_sign() {
    for value in [-9.0, -1.0] {
        assert!(extract_invoice(vec![total_envelope(value, &["Invoice.total"])], "Invoice from Acme Corp, $1,200.00, due 2026-09-01").is_err());
    }
}

#[test]
fn extract_does_not_read_a_range_hyphen_as_a_minus_sign() {
    assert!(extract_invoice(vec![total_envelope(-1400.0, &["Invoice.total"])], "Invoice from Acme Corp, $1,200.00-1,400.00, due 2026-09-01").is_err());
}

#[test]
fn extract_clears_a_written_negative() {
    let result = extract_invoice(vec![total_envelope(-1200.0, &["Invoice.total"])], "Invoice from Acme Corp, -1,200.00, due 2026-09-01").unwrap().unwrap();
    assert_eq!(result["total"], -1200.0);
}

#[test]
fn extract_clears_a_zero_against_its_source() {
    let result = extract_invoice(vec![total_envelope(0.0, &["Invoice.total"])], "Invoice from Acme Corp, $0.00, due 2026-09-01").unwrap().unwrap();
    assert_eq!(result["total"], 0.0);
}

fn thermostat_agent(envelopes: Vec<rustyneedle_needle::Response>) -> Needle {
    let (engine, _) = Stub::new(envelopes);
    Needle::builder()
        .engine(engine)
        .tool_with_handler(thermostat_schema(), |args| {
            Ok(json!({"temperature": args["temperature"], "mode": args.get("mode").cloned().unwrap_or(json!("auto"))}))
        })
        .clock(clock())
        .build()
}

#[test]
fn run_executes_a_grounded_number_and_refuses_an_ungrounded_sibling() {
    let mut first = envelope("set_thermostat", json!({"temperature": 21}), &["set_thermostat.temperature"]);
    first.function_calls.push(Call { name: "set_thermostat".into(), arguments: arguments(json!({"temperature": 22})) });
    let mut agent = thermostat_agent(vec![first, finished()]);
    let response = agent.run("make it 21 and cool the room", &RunOptions::default());
    assert_eq!(response.results.unwrap(), vec![json!({"temperature": 21, "mode": "auto"}), json!({"error": "ungrounded temperature"})]);
}

#[test]
fn run_executes_a_separated_number_grounded_in_the_query() {
    let mut agent = thermostat_agent(vec![envelope("set_thermostat", json!({"temperature": 1200}), &["set_thermostat.temperature"]), finished()]);
    let response = agent.run("set it to 1,200", &RunOptions::default());
    assert_eq!(response.results.unwrap(), vec![json!({"temperature": 1200, "mode": "auto"})]);
}

#[test]
fn run_refuses_a_number_absent_from_the_query() {
    let mut agent = thermostat_agent(vec![envelope("set_thermostat", json!({"temperature": 99}), &["set_thermostat.temperature"]), finished()]);
    let response = agent.run("make it 21 and cool the room", &RunOptions::default());
    assert_eq!(response.results.unwrap(), vec![json!({"error": "ungrounded temperature"})]);
}

#[test]
fn run_still_refuses_a_non_numeric_engine_flag() {
    let mut agent = thermostat_agent(vec![envelope("set_thermostat", json!({"temperature": 21}), &["set_thermostat.mode"]), finished()]);
    let response = agent.run("make it 21 and cool the room", &RunOptions::default());
    assert_eq!(response.results.unwrap(), vec![json!({"error": "ungrounded mode"})]);
}

#[test]
fn a_time_after_a_day_and_month_is_not_read_as_a_year() {
    let (engine, _) = Stub::new(vec![dated_call("2026-06-05")]);
    let response = invoice_agent(engine, "").complete("Send an invoice to Acme due on 5 June 19:30");
    assert!(response.validation.is_none());
    assert!(source_years("due on 5 June 19:30").is_empty());
    assert!(source_years("due on June 5 7 pm").is_empty());
    assert_eq!(source_years("due on June 5, 2031 19:30"), years(&[2031]));
    assert_eq!(source_years("due on 5 June 2031"), years(&[2031]));
}

#[test]
fn extract_accepts_a_date_followed_by_a_time() {
    let result = extract_invoice(vec![dated_call("2026-03-03")], "Invoice from Acme Corp due March 3 14:00").unwrap().unwrap();
    assert_eq!(result["due_date"], "2026-03-03");
}

#[test]
fn an_abbreviated_month_with_a_period_still_gives_its_year() {
    for (text, expected) in
        [("due Dec. 25, 2026", 2026), ("due 25 Dec. 2026", 2026), ("due Sept. 12, 2025", 2025), ("due 12 Sept. 2025", 2025), ("due Sep. 2025", 2025)]
    {
        assert_eq!(source_years(text), years(&[expected]), "{text}");
    }
}

#[test]
fn extract_accepts_dates_written_with_abbreviated_months_and_periods() {
    let result = extract_invoice(vec![dated_call("2026-12-25")], "Invoice from Acme, issued Jun 5, 2025 and due Dec. 25, 2026").unwrap().unwrap();
    assert_eq!(result["due_date"], "2026-12-25");
}
