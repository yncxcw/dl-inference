#include <string>
#include <string_view>
#include <vector>

#include "dli/logging.h"
#include "test_support.h"

int main() {
  return dli_test::run("LogMessage", [] {
    dli_test::expect(dli::logLevel() == dli::LogSeverity::Warn, "default logging level is WARN");

    struct Record {
      dli::LogSeverity severity;
      std::string message;
    };
    std::vector<Record> records;
    dli::setLogSink(
        [&](dli::LogSeverity severity, std::string_view, int, std::string_view message) {
          records.push_back({severity, std::string(message)});
        });
    const auto previous_level = dli::logLevel();
    dli::setLogLevel(dli::LogSeverity::Debug);
    LOG_DEBUG << "hello "
              << "debug";
    LOG_INFO << "hello "
             << "info";
    LOG_WARN << "hello "
             << "warn";
    LOG_ERROR << "hello "
              << "error";
    dli::setLogLevel(previous_level);
    dli::resetLogSink();

    dli_test::expect(records.size() == 4, "logging record count");
    dli_test::expect(records[0].severity == dli::LogSeverity::Debug, "debug severity");
    dli_test::expect(records[0].message == "hello debug", "debug message");
    dli_test::expect(records[3].severity == dli::LogSeverity::Error, "error severity");
    dli_test::expect(records[3].message == "hello error", "error message");

    records.clear();
    dli::setLogSink(
        [&](dli::LogSeverity severity, std::string_view, int, std::string_view message) {
          records.push_back({severity, std::string(message)});
        });
    dli::setLogLevel(dli::LogSeverity::Warn);
    LOG_DEBUG << "debug";
    LOG_INFO << "info";
    LOG_WARN << "warn";
    LOG_ERROR << "error";
    dli::setLogLevel(previous_level);
    dli::resetLogSink();

    dli_test::expect(records.size() == 2, "logging severity level record count");
    dli_test::expect(records[0].severity == dli::LogSeverity::Warn, "warn threshold keeps warn");
    dli_test::expect(records[0].message == "warn", "warn threshold message");
    dli_test::expect(records[1].severity == dli::LogSeverity::Error, "warn threshold keeps error");
    dli_test::expect(records[1].message == "error", "error threshold message");
  });
}
