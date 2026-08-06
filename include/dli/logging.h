#pragma once

#include <functional>
#include <iosfwd>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

namespace dli {

enum class LogSeverity { Debug, Info, Warn, Error };

using LogSink = std::function<void(LogSeverity severity, std::string_view file, int line,
                                   std::string_view message)>;

const char* toString(LogSeverity severity);
void setLogLevel(LogSeverity severity);
LogSeverity logLevel();
void setLogSink(LogSink sink);
void resetLogSink();

class LogMessage {
 public:
  LogMessage(LogSeverity severity, const char* file, int line);
  LogMessage(const LogMessage&) = delete;
  LogMessage& operator=(const LogMessage&) = delete;
  ~LogMessage() noexcept;

  std::ostream& stream() { return stream_; }

 private:
  LogSeverity severity_;
  const char* file_;
  int line_;
  std::ostringstream stream_;
};

}  // namespace dli

#define LOG_DEBUG ::dli::LogMessage(::dli::LogSeverity::Debug, __FILE__, __LINE__).stream()
#define LOG_INFO ::dli::LogMessage(::dli::LogSeverity::Info, __FILE__, __LINE__).stream()
#define LOG_WARN ::dli::LogMessage(::dli::LogSeverity::Warn, __FILE__, __LINE__).stream()
#define LOG_ERROR ::dli::LogMessage(::dli::LogSeverity::Error, __FILE__, __LINE__).stream()
