# Writing tools that take dates and times

## The date fact

By default the agent puts a line such as `date: 2026-10-22 Thu 13:20` at the start of the system prompt, as the original library does. The model has no clock, so this is how it learns today's date. The grounding check also uses it: a year in a date argument is accepted when the request or the system facts support it, and flagged in `validation.ungrounded` otherwise.

Two consequences follow.

- The line changes every minute, and it is fixed when the agent is built. A cached prompt prefix is valid for one minute, and a long-lived agent keeps the date it was built with.
- It changes results even for requests that mention no date. On the original's six environments (192 cases) it changed 4 results; one of them was worse with the fact than without.

## Let the host resolve dates and times

The original project's own tools do not ask the model to compute dates. In its productivity environment the parameters are strings named `time_human`, `date_time_human` and `start_time_human`, documented as "the user's date or time phrase copied word for word", and the system prompt says "copy titles, messages, and date or time phrases verbatim from the user; never rephrase or resolve them". The application turns "in 5 minutes", "Friday at 3pm" or "March 4th" into a timestamp with its own clock.

With tools written that way the date fact is not needed: "remind me in an hour" returns `in an hour`, "dentist on Friday at 3pm" returns `Friday at 3pm`. Pass `--no-date` (or `Needle::builder().auto_date(false)`) and the prompt prefix stays the same all day.

## When the model has to produce an ISO date

If a parameter must be `YYYY-MM-DD` or `HH:MM`, keep the date fact (the default) so weekday names and month-and-day dates resolve. Be aware of the limits:

- Weekday names and month-and-day dates resolve, "March 4th" takes the current year.
- Phrases that need the time of day, such as "in 5 minutes" or "in an hour", are not turned into a clock time even with the minute in the fact; they come back as the phrase.
- A request that gives a time but no date ("at 3pm") gets today's date with the fact, and a made-up date without it.

For durations, use an integer parameter (`minutes: 5`, as Home Assistant's timers do). Those are copied from the request and do not depend on the date.

## Triggers

Some requests must reach a tool even when the model hesitates. A tool may carry `"triggers"`, a list of case-insensitive regular expressions. A request that matches one is routed to that tool and the output is required to be a call; the call ships even when its confidence is low, so the confidence score tells you whether to run it or ask first. In `examples/tools/weather_and_time.json`, "what time is it in Tokyo" is answered with `get_weather` without the trigger on `get_local_time`; with it, the right tool is called. The same file pattern fixes `call_contact` in `car.json` and `talk_to_human` in `customer_support.json`.

From Rust, `tool("name").triggers([r"\bcall\b"])` sets them.

## Tool sets to start from

`examples/tools/` and `examples/environments/` hold eleven tool sets (see [examples/README.md](../examples/README.md)). Keep a set to five tools or fewer: Needle 3 reads every tool in the prompt, has no retrieval stage and a context of 8,192 tokens, so larger sets lose accuracy. Prefer enums for closed sets, bounded integers, plain names without a shared prefix, and verbatim copy for free text.
