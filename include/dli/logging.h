/**
 * @file logging.h
 * @brief Minimal thread-safe logging with a severity threshold and pluggable sink.
 */

#ifndef DLI_LOGGING_H_
#define DLI_LOGGING_H_

#include <functional>
#include <iosfwd>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

namespace dli {

/// Message severity, in increasing order.
enum class LogSeverity { Debug, Info, Warn, Error };

/**
 * @brief Receives each emitted log record.
 *
 * Arguments are the severity, source file, source line and formatted message. The sink is invoked
 * while an internal mutex is held, so it must not log.
 */
using LogSink = std::function<void(LogSeverity severity, std::string_view file, int line,
                                   std::string_view message)>;

/**
 * @brief Returns the upper-case name of a severity.
 * @param severity Severity to name.
 * @return A static string such as "WARN".
 */
const char* toString(LogSeverity severity);

/**
 * @brief Sets the minimum severity that is emitted (default Warn).
 * @param severity New threshold.
 */
void setLogLevel(LogSeverity severity);

/// @return The current minimum severity.
LogSeverity logLevel();

/**
 * @brief Replaces the destination of log records.
 * @param sink New sink; an empty function restores the default stderr sink.
 */
void setLogSink(LogSink sink);

/// Restores the default stderr sink.
void resetLogSink();

/**
 * @brief Builds one log record and emits it on destruction.
 *
 * Normally used through the LOG_* macros rather than directly.
 */
class LogMessage {
 public:
  /**
   * @brief Starts a record.
   * @param severity Record severity.
   * @param file Source file name; must outlive this object.
   * @param line Source line.
   */
  LogMessage(LogSeverity severity, const char* file, int line);
  LogMessage(const LogMessage&) = delete;
  LogMessage& operator=(const LogMessage&) = delete;

  /// Emits the record to the sink if its severity meets the threshold; never throws.
  ~LogMessage() noexcept;

  /// @return Stream the message text is written to.
  std::ostream& stream() { return stream_; }

 private:
  /// Record severity.
  LogSeverity severity_;
  /// Source file name (borrowed).
  const char* file_;
  /// Source line.
  int line_;
  /// Accumulated message text.
  std::ostringstream stream_;
};

}  // namespace dli

/// Streams a Debug record, e.g. `LOG_DEBUG << "x=" << x;`.
#define LOG_DEBUG ::dli::LogMessage(::dli::LogSeverity::Debug, __FILE__, __LINE__).stream()
/// Streams an Info record.
#define LOG_INFO ::dli::LogMessage(::dli::LogSeverity::Info, __FILE__, __LINE__).stream()
/// Streams a Warn record.
#define LOG_WARN ::dli::LogMessage(::dli::LogSeverity::Warn, __FILE__, __LINE__).stream()
/// Streams an Error record.
#define LOG_ERROR ::dli::LogMessage(::dli::LogSeverity::Error, __FILE__, __LINE__).stream()

#endif  // DLI_LOGGING_H_
