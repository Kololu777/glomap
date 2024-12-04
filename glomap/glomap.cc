#include "glomap/exe/global_mapper.h"

#include <iostream>
#include <span>
#include <sstream>
#include <string_view>
#include <unordered_map>

namespace cli {

    struct CommandLine {
        std::span<char*> args;

        explicit CommandLine(int argc, char** argv)
            : args(argv, argc) {}

        [[nodiscard]] std::string_view command() const {
            return args.size() > 1 ? args[1] : "";
        }

        [[nodiscard]] auto commandArgs() const {
            if (args.size() <= 1)
                return std::span<char*>{};
            auto cmdArgs = args.subspan(1);
            cmdArgs[0] = args[0]; // Replace command name with program name
            return cmdArgs;
        }
    };

    class CommandRegistry {
    public:
        using CommandFunc = std::function<int(int, char**)>;

        void registerCommand(std::string_view name, CommandFunc func) {
            commands_.emplace(name, std::move(func));
        }

        [[nodiscard]] auto findCommand(std::string_view name) const -> const CommandFunc* {
            if (auto it = commands_.find(name); it != commands_.end())
            {
                return &it->second;
            }
            return nullptr;
        }

        [[nodiscard]] auto getCommands() const -> const auto& {
            return commands_;
        }

    private:
        std::unordered_map<std::string_view, CommandFunc> commands_;
    };

    [[nodiscard]] int showHelp(const CommandRegistry& registry) {
        constexpr auto usage = R"(
GLOMAP -- Global Structure-from-Motion

Usage:
  glomap mapper --database_path DATABASE --output_path MODEL
  glomap mapper_resume --input_path MODEL_INPUT --output_path MODEL_OUTPUT

Available commands:
  help)";

        std::cout << usage << '\n';

        for (const auto& [name, _] : registry.getCommands())
        {
            std::cout << "  " << name << '\n';
        }

        std::cout << '\n';
        return EXIT_SUCCESS;
    }

    [[nodiscard]] bool isHelpCommand(std::string_view cmd) {
        return cmd == "help" || cmd == "-h" || cmd == "--help";
    }

    [[nodiscard]] std::string makeErrorMessage(std::string_view command) {
        std::ostringstream oss;
        oss << "Command '" << command << "' not recognized. "
            << "Run 'glomap help' to list available commands.\n";
        return oss.str();
    }

} // namespace cli

int main(int argc, char** argv) {
    glomap::InitializeGlog(argv);
    FLAGS_alsologtostderr = true;

    cli::CommandRegistry registry;
    registry.registerCommand("mapper", &glomap::RunMapper);
    registry.registerCommand("mapper_resume", &glomap::RunMapperResume);

    const cli::CommandLine cmdLine(argc, argv);

    if (cmdLine.args.empty() || cli::isHelpCommand(cmdLine.command()))
    {
        return showHelp(registry);
    }

    if (const auto* cmd = registry.findCommand(cmdLine.command()))
    {
        auto args = cmdLine.commandArgs();
        return (*cmd)(args.size(), args.data());
    }

    std::cerr << cli::makeErrorMessage(cmdLine.command());
    return EXIT_FAILURE;
}