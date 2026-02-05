#ifndef NPZ_WRITER_HPP
#define NPZ_WRITER_HPP

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>
#include "NPYWriter.hpp"

/**
 * @brief Utility for writing NumPy .npz format (zipped collection of .npy files)
 *
 * NPZ format is a ZIP archive containing multiple .npy files.
 * This implementation uses system zip command for simplicity.
 */
class NPZWriter
{
public:
    /**
     * @brief Start a new NPZ file (creates temp directory)
     * @param filename Output .npz file path
     */
    static void begin(const std::string &filename)
    {
        current_file = filename;
        temp_dir = ".npz_temp_" + std::to_string(getpid());

        // Create temp directory
        mkdir(temp_dir.c_str(), 0755);
        array_files.clear();
    }

    /**
     * @brief Add a 1D double array to the current NPZ
     * @param name Array name (without .npy extension)
     * @param data Vector of doubles
     */
    static void add_array(const std::string &name, const std::vector<double> &data)
    {
        std::string filepath = temp_dir + "/" + name + ".npy";
        NPYWriter::write_array_1d(filepath, data);
        array_files.push_back(filepath);
    }

    /**
     * @brief Add a 1D int array to the current NPZ
     * @param name Array name (without .npy extension)
     * @param data Vector of integers
     */
    static void add_array(const std::string &name, const std::vector<int> &data)
    {
        std::string filepath = temp_dir + "/" + name + ".npy";
        NPYWriter::write_array_1d(filepath, data);
        array_files.push_back(filepath);
    }

    /**
     * @brief Add a 1D uint array to the current NPZ
     * @param name Array name (without .npy extension)
     * @param data Vector of unsigned integers
     */
    static void add_array(const std::string &name, const std::vector<unsigned int> &data)
    {
        std::string filepath = temp_dir + "/" + name + ".npy";
        NPYWriter::write_array_1d(filepath, data);
        array_files.push_back(filepath);
    }

    /**
     * @brief Add a 2D double array to the current NPZ
     * @param name Array name (without .npy extension)
     * @param data Flattened 2D array (row-major)
     * @param rows Number of rows
     * @param cols Number of columns
     */
    static void add_array_2d(const std::string &name, const std::vector<double> &data,
                             size_t rows, size_t cols)
    {
        std::string filepath = temp_dir + "/" + name + ".npy";
        NPYWriter::write_array_2d(filepath, data, rows, cols);
        array_files.push_back(filepath);
    }

    /**
     * @brief Finalize and write the NPZ file
     * Zips all added arrays and cleans up temp files
     */
    static void finalize()
    {
        if (array_files.empty())
        {
            return;
        }

        // Build zip command
        std::ostringstream cmd;
        cmd << "cd " << temp_dir << " && zip -q -r ../" << current_file << " *.npy";

        int ret = system(cmd.str().c_str());
        if (ret != 0)
        {
            std::cerr << "Warning: zip command failed for " << current_file << std::endl;
        }

        // Clean up temp files
        for (const auto &file : array_files)
        {
            unlink(file.c_str());
        }
        rmdir(temp_dir.c_str());

        array_files.clear();
    }

private:
    static std::string current_file;
    static std::string temp_dir;
    static std::vector<std::string> array_files;
};

// Static member initialization
std::string NPZWriter::current_file;
std::string NPZWriter::temp_dir;
std::vector<std::string> NPZWriter::array_files;

#endif // NPZ_WRITER_HPP
