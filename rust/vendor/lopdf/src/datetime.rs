use super::Object;

#[cfg(feature = "chrono")]
mod chrono_impl {
    use crate::{Object, datetime::convert_utc_offset};
    use chrono::prelude::*;

    impl From<DateTime<FixedOffset>> for Object {
        fn from(date: DateTime<FixedOffset>) -> Self {
            let mut timezone_str = date.format("D:%Y%m%d%H%M%S%:z'").to_string().into_bytes();
            convert_utc_offset(&mut timezone_str);
            Object::string_literal(timezone_str)
        }
    }

    impl From<DateTime<Utc>> for Object {
        fn from(date: DateTime<Utc>) -> Self {
            Object::string_literal(date.format("D:%Y%m%d%H%M%SZ").to_string())
        }
    }

    /// A PDF date states a fixed offset from UT and never a named zone
    /// (ISO 32000-1, 7.9.4), so this is the conversion that keeps what the
    /// file said. Re-expressing it in the machine's own zone is what the
    /// `chrono-clock` feature adds.
    impl TryFrom<super::DateTime> for DateTime<FixedOffset> {
        type Error = chrono::format::ParseError;

        fn try_from(value: super::DateTime) -> Result<DateTime<FixedOffset>, Self::Error> {
            let from_date = |date: NaiveDate| {
                FixedOffset::east_opt(0)
                    .unwrap()
                    .from_utc_datetime(&date.and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap()))
            };

            DateTime::parse_from_str(&value.0, "%Y%m%d%H%M%S%#z")
                .or_else(|_| DateTime::parse_from_str(&value.0, "%Y%m%d%H%M%#z"))
                .or_else(|_| NaiveDate::parse_from_str(&value.0, "%Y%m%d").map(from_date))
        }
    }
}

/// The system-local conversions, kept apart from `chrono` because `Local`
/// exists only with chrono's `clock` feature — which brings `iana-time-zone`
/// and its per-platform chain. A PDF date never names a zone, so a consumer
/// that does not want the reading machine's offset applied to someone else's
/// document can have dates for `chrono` and `num-traits` alone.
#[cfg(feature = "chrono-clock")]
mod chrono_clock_impl {
    use crate::{Object, datetime::convert_utc_offset};
    use chrono::prelude::*;

    impl From<DateTime<Local>> for Object {
        fn from(date: DateTime<Local>) -> Self {
            let mut timezone_str = date.format("D:%Y%m%d%H%M%S%:z'").to_string().into_bytes();
            convert_utc_offset(&mut timezone_str);
            Object::string_literal(timezone_str)
        }
    }

    impl TryFrom<super::DateTime> for DateTime<Local> {
        type Error = chrono::format::ParseError;

        fn try_from(value: super::DateTime) -> Result<DateTime<Local>, Self::Error> {
            DateTime::<FixedOffset>::try_from(value).map(|date| date.with_timezone(&Local))
        }
    }
}

#[cfg(feature = "jiff")]
mod jiff_impl {
    use crate::{Object, datetime::convert_utc_offset};
    use jiff::{Timestamp, Zoned};

    impl From<Zoned> for Object {
        fn from(date: Zoned) -> Self {
            let mut timezone_str = date.strftime("D:%Y%m%d%H%M%S%:z'").to_string().into_bytes();
            convert_utc_offset(&mut timezone_str);
            Object::string_literal(timezone_str)
        }
    }

    impl From<Timestamp> for Object {
        fn from(date: Timestamp) -> Self {
            Object::string_literal(date.strftime("D:%Y%m%d%H%M%SZ").to_string())
        }
    }

    impl TryFrom<super::DateTime> for Zoned {
        type Error = jiff::Error;

        fn try_from(value: super::DateTime) -> Result<Self, Self::Error> {
            use jiff::civil::{Date, DateTime};

            // We attempt to parse different date time formats based on Section 7.9.4 "Dates" in
            // PDF 32000-1:2008 here.
            //
            // "A PLUS SIGN as the value of the O field signifies that the local time is later than
            // UT, a HYPHEN-MINUS signifies that local time is earlier than UT, and the LATIN
            // CAPITAL Z signifies that local time is equal to UT. If no UT information is
            // specified, the relationship of the specified time to UT shall be considered GMT."
            //
            // 1. Try parsing the full date and time with the `%#z` specifier to parse the timezone as a `Zoned` object.
            // 2. Try parsing the full date and time with the 'Z' suffix as a `DateTime` interpreted to be in the UTC
            //    timezone.
            // 3. Try parsing the date and time without the seconds specified with the `%#z` specifier to parse the
            //    timezone as a `Zoned` object.
            // 4. Try parsing the date and time without the seconds specified with the 'Z' as a `DateTime` interpreted
            //    to be in the UTC timezone.
            // 5. Try parsing the date with no time as a `Date` interpreted to be in the GMT timezone.
            //
            // In all cases we return a `Zoned` object here to preserve the timezone.
            Zoned::strptime("%Y%m%d%H%M%S%#z", &value.0)
                .or_else(|_| DateTime::strptime("%Y%m%d%H%M%SZ", &value.0).and_then(|dt| dt.in_tz("UTC")))
                .or_else(|_| Zoned::strptime("%Y%m%d%H%M%#z", &value.0))
                .or_else(|_| DateTime::strptime("%Y%m%d%H%MZ", &value.0).and_then(|dt| dt.in_tz("UTC")))
                .or_else(|_| Date::strptime("%Y%m%d", &value.0).and_then(|dt| dt.at(0, 0, 0, 0).in_tz("GMT")))
        }
    }
}

#[cfg(feature = "time")]
mod time_impl {
    use crate::Object;
    use time::{OffsetDateTime, PrimitiveDateTime};

    /// The naive datetime is taken to be UTC and rendered with the `Z` suffix
    /// (PDF 32000-1 §7.9.4) — the `time`-crate counterpart of the
    /// `chrono::DateTime<Utc>` and `jiff::Timestamp` impls above.
    ///
    /// (This replaces a `From<time::Time>` impl that never compiled — see
    /// issue #518: it referenced a nonexistent `FormatItem::StringLiteral`
    /// with a strftime pattern, and a bare time-of-day cannot form a valid
    /// PDF date anyway, which must start at the year.)
    impl From<PrimitiveDateTime> for Object {
        fn from(date: PrimitiveDateTime) -> Self {
            Object::string_literal({
                // D:%Y%m%d%H%M%SZ
                let format =
                    time::format_description::parse_borrowed::<3>("D:[year][month][day][hour][minute][second]Z")
                        .unwrap();
                date.format(&format).unwrap()
            })
        }
    }

    impl From<OffsetDateTime> for Object {
        fn from(date: OffsetDateTime) -> Self {
            Object::string_literal({
                // D:%Y%m%d%H%M%S:%z'
                let format = time::format_description::parse_borrowed::<3>(
                    "D:[year][month][day][hour][minute][second][offset_hour sign:mandatory]'[offset_minute]'",
                )
                .unwrap();
                date.format(&format).unwrap()
            })
        }
    }

    /// WARNING: `tm_wday` (weekday), `tm_yday` (day index in year), `tm_isdst`
    /// (daylight saving time) and `tm_nsec` (nanoseconds of the date from 1970)
    /// are set to 0 since they aren't available in the PDF time format. They could,
    /// however, be calculated manually
    impl TryFrom<super::DateTime> for OffsetDateTime {
        type Error = time::Error;

        fn try_from(value: super::DateTime) -> Result<OffsetDateTime, Self::Error> {
            let format = time::format_description::parse_borrowed::<3>(
                "[year][month][day][hour][minute][second][offset_hour sign:mandatory][offset_minute]",
            )
            .unwrap();

            Ok(OffsetDateTime::parse(&value.0, &format)?)
        }
    }
}

// Find the last `:` and turn it into an `'` to account for PDF weirdness
#[allow(dead_code)]
fn convert_utc_offset(bytes: &mut [u8]) {
    let mut index = bytes.len();
    while let Some(last) = bytes[..index].last_mut() {
        if *last == b':' {
            *last = b'\'';
            break;
        }
        index -= 1;
    }
}

#[derive(Clone, Debug)]
pub struct DateTime(String);

impl Object {
    // Parses the `D`, `:` and `\` out of a `Object::String` to parse the date time
    fn datetime_string(&self) -> Option<String> {
        if let Object::String(bytes, _) = self {
            String::from_utf8(bytes.iter().filter(|b| !b"D:'".contains(b)).cloned().collect()).ok()
        } else {
            None
        }
    }

    pub fn as_datetime(&self) -> Option<DateTime> {
        self.datetime_string().map(DateTime)
    }
}

impl DateTime {
    /// The date's characters with `D`, `:` and `'` removed.
    ///
    /// Every typed conversion of this value needs a date backend — `chrono`,
    /// `jiff` or `time` — but [`Object::as_datetime`] itself needs none. A
    /// build that enables no backend could therefore obtain a `DateTime` and
    /// have no way to read it. This is that way.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[test]
fn a_datetime_is_readable_without_a_backend() {
    let text = Object::string_literal("D:20260710143000+02'00'");
    // The `D`, `:` and apostrophes are filtered out on the way in; the
    // relation sign survives, because it is what the offset means.
    assert_eq!(text.as_datetime().unwrap().as_str(), "20260710143000+0200");
    assert!(Object::Integer(20260710).as_datetime().is_none());
}

#[cfg(feature = "chrono")]
#[test]
fn parse_datetime_fixed_offset_keeps_the_offset_the_file_stated() {
    use chrono::prelude::*;

    let text = Object::string_literal("D:20260710143000+02'00'");
    let date: DateTime<FixedOffset> = text.as_datetime().unwrap().try_into().unwrap();
    // Not shifted into UTC, and not into the reading machine's zone either.
    assert_eq!(date.offset().local_minus_utc(), 2 * 3600);
    assert_eq!(date.hour(), 14);
    // And it round-trips back to the same text.
    let back: Object = date.into();
    assert_eq!(back, Object::string_literal("D:20260710143000+02'00'"));
}

#[cfg(feature = "chrono")]
#[test]
fn parse_datetime_fixed_offset_round_trip() {
    use chrono::prelude::*;

    // A fixed instant rather than `now()`: `Utc::now` needs chrono's `clock`
    // feature, which this row deliberately does without, and a deterministic
    // test is the better one regardless.
    let time = FixedOffset::east_opt(-5 * 3600 - 30 * 60)
        .unwrap()
        .with_ymd_and_hms(2026, 7, 10, 14, 30, 0)
        .unwrap();
    let text: Object = time.into();
    let time2: Option<DateTime<FixedOffset>> = text.as_datetime().and_then(|dt| dt.try_into().ok());
    assert_eq!(time2, Some(time));
}

#[cfg(feature = "chrono-clock")]
#[test]
fn parse_datetime_local() {
    use chrono::prelude::*;

    let time = Local::now().with_nanosecond(0).unwrap();
    let text: Object = time.into();
    let time2: Option<DateTime<Local>> = text.as_datetime().and_then(|dt| dt.try_into().ok());
    assert_eq!(time2, Some(time));
}

#[cfg(feature = "chrono")]
#[test]
fn parse_datetime_utc() {
    use chrono::prelude::*;

    let time = Utc.with_ymd_and_hms(2026, 7, 10, 14, 30, 0).unwrap();
    let text: Object = time.into();
    let time2: Option<DateTime<FixedOffset>> = text.as_datetime().and_then(|dt| dt.try_into().ok());
    assert_eq!(time2, Some(time.fixed_offset()));
}

#[cfg(feature = "jiff")]
#[test]
fn parse_zoned() {
    use jiff::Zoned;

    let time = Zoned::now().with().subsec_nanosecond(0).build().unwrap();
    let text: Object = time.clone().into();
    let time2: Option<Zoned> = text.as_datetime().and_then(|dt| dt.try_into().ok());
    assert_eq!(time2, Some(time));
}

#[cfg(feature = "jiff")]
#[test]
fn parse_timestamp() {
    use jiff::Zoned;

    let time = Zoned::now().with().subsec_nanosecond(0).build().unwrap();
    let text: Object = time.timestamp().into();
    let time2: Option<Zoned> = text.as_datetime().and_then(|dt| dt.try_into().ok());
    assert_eq!(time2, Some(time));
}

#[cfg(feature = "chrono")]
#[test]
fn parse_datetime_seconds_missing_chrono() {
    use chrono::prelude::*;

    // this is the example from the PDF reference, version 1.7, chapter 3.8.3
    let text = Object::string_literal("D:199812231952-08'00'");
    let dt: Option<DateTime<FixedOffset>> = text.as_datetime().and_then(|dt| dt.try_into().ok());
    assert!(dt.is_some());
}

#[cfg(feature = "chrono")]
#[test]
fn parse_datetime_time_missing_chrono() {
    use chrono::prelude::*;

    let text = Object::string_literal("D:20040229");
    let dt: Option<DateTime<FixedOffset>> = text.as_datetime().and_then(|dt| dt.try_into().ok());
    assert!(dt.is_some());
}

#[cfg(feature = "jiff")]
#[test]
fn parse_datetime_seconds_missing_jiff() {
    use jiff::Zoned;

    // this is the example from the PDF reference, version 1.7, chapter 3.8.3
    let text = Object::string_literal("D:199812231952-08'00'");
    let dt: Option<Zoned> = text.as_datetime().and_then(|dt| dt.try_into().ok());
    assert!(dt.is_some());
}

#[cfg(feature = "jiff")]
#[test]
fn parse_datetime_time_missing_jiff() {
    use jiff::Zoned;

    let text = Object::string_literal("D:20040229");
    let dt: Option<Zoned> = text.as_datetime().and_then(|dt| dt.try_into().ok());
    assert!(dt.is_some());
}

#[cfg(feature = "time")]
#[test]
fn format_primitive_datetime_as_utc() {
    use time::{Date, Month, PrimitiveDateTime, Time};

    // The example date from the PDF reference (1.7, §3.8.3), as a naive
    // datetime: it must serialize with the year first and the `Z` suffix.
    let date = Date::from_calendar_date(1998, Month::December, 23).unwrap();
    let time = Time::from_hms(19, 52, 0).unwrap();
    let text: Object = PrimitiveDateTime::new(date, time).into();
    match &text {
        Object::String(bytes, _) => assert_eq!(bytes.as_slice(), b"D:19981223195200Z"),
        other => panic!("expected a string literal, got {other:?}"),
    }
}

#[cfg(feature = "time")]
#[test]
fn parse_datetime() {
    use time::OffsetDateTime;

    let time = OffsetDateTime::now_utc();

    let text: Object = time.into();
    let time2: OffsetDateTime = text.as_datetime().unwrap().try_into().unwrap();

    assert_eq!(time2.date(), time.date());

    // Ignore nanoseconds
    // - not important in the date parsing
    assert_eq!(time2.time().hour(), time.time().hour());
    assert_eq!(time2.time().minute(), time.time().minute());
    assert_eq!(time2.time().second(), time.time().second());
}
