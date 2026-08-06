#include "dli/logging.h"

#include <iostream>
#include <mutex>
#include <utility>

namespace dli {
namespace {

std::mutex& sinkMutex() {
  static std::mutex mutex;
  return mutex;
}

LogSink& sinkStorage() {
  static LogSink sink;
  return sink;
}

LogSeverity& logLevelStorage() {
  static LogSeverity level = LogSeverity::Warn;
  return level;
}

int severityRank(LogSeverity severity) {
  switch (severity) {
    case LogSeverity::Debug:
      return 0;
    case LogSeverity::Info:
      return 1;
    case LogSeverity::Warn:
      return 2;
    case LogSeverity::Error:
      return 3;
  }
  return 0;
}

void defaultSink(LogSeverity severity, std::string_view file, int line, std::string_view message) {
  std::cerr << "[" << toString(severity) << "] " << file << ":" << line << " " << message << "\n";
}

}  // namespace

const char* toString(LogSeverity severity) {
  switch (severity) {
    case LogSeverity::Debug:
      return "DEBUG";
    case LogSeverity::Info:
      return "INFO";
    case LogSeverity::Warn:
      return "WARN";
    case LogSeverity::Error:
      return "ERROR";
  }
  return "UNKNOWN";
}

void setLogLevel(LogSeverity severity) {
  std::lock_guard<std::mutex> lock(sinkMutex());
  logLevelStorage() = severity;
}

LogSeverity logLevel() {
  std::lock_guard<std::mutex> lock(sinkMutex());
  return logLevelStorage();
}

void setLogSink(LogSink sink) {
  std::lock_guard<std::mutex> lock(sinkMutex());
  sinkStorage() = std::move(sink);
}

void resetLogSink() { setLogSink({}); }

LogMessage::LogMessage(LogSeverity severity, const char* file, int line)
    : severity_(severity), file_(file), line_(line) {}

LogMessage::~LogMessage() noexcept {
  try {
    const auto message = stream_.str();
    std::lock_guard<std::mutex> lock(sinkMutex());
    if (severityRank(severity_) < severityRank(logLevelStorage())) return;
    if (sinkStorage()) {
      sinkStorage()(severity_, file_, line_, message);
    } else {
      defaultSink(severity_, file_, line_, message);
    }
  } catch (...) {
  }
}

}  // namespace dli
