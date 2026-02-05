#ifndef NPY_WRITER_HPP
#define NPY_WRITER_HPP

#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <sstream>
#include <iomanip>

/**
 * @brief Utility class for writing NumPy .npy format files
 *
 * Implements NumPy .npy format version 1.0 specification:
 * - Magic string: "\x93NUMPY"
 * - Version: 1.0
 * - Header: Python dict with dtype, shape, fortran_order
 * - Data: Binary array data in C order (row-major)
 */
class NPYWriter
{
public:
    /**
     * @brief Write a 1D double array to .npy file
     * @param filename Output file path
     * @param data Vector of doubles
     * @param append If true, append to existing file (for multiple arrays)
     */
    static void write_array_1d(const std::string &filename, const std::vector<double> &data, bool append = false)
    {
        write_array_1d_impl(filename, data.data(), data.size(), append);
    }

    /**
     * @brief Write a 1D integer array to .npy file
     * @param filename Output file path
     * @param data Vector of integers
     * @param append If true, append to existing file
     */
    static void write_array_1d(const std::string &filename, const std::vector<int> &data, bool append = false)
    {
        write_array_1d_int_impl(filename, data.data(), data.size(), append);
    }

    /**
     * @brief Write a 1D uint array to .npy file
     * @param filename Output file path
     * @param data Vector of unsigned integers
     * @param append If true, append to existing file
     */
    static void write_array_1d(const std::string &filename, const std::vector<unsigned int> &data, bool append = false)
    {
        write_array_1d_uint_impl(filename, data.data(), data.size(), append);
    }

    /**
     * @brief Write a 2D double array to .npy file
     * @param filename Output file path
     * @param data Flattened 2D array (row-major order)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param append If true, append to existing file
     */
    static void write_array_2d(const std::string &filename, const std::vector<double> &data,
                                size_t rows, size_t cols, bool append = false)
    {
        write_array_2d_impl(filename, data.data(), rows, cols, append);
    }

    /**
     * @brief Write a 2D integer array to .npy file
     * @param filename Output file path
     * @param data Flattened 2D array (row-major order)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param append If true, append to existing file
     */
    static void write_array_2d(const std::string &filename, const std::vector<int> &data,
                                size_t rows, size_t cols, bool append = false)
    {
        write_array_2d_int_impl(filename, data.data(), rows, cols, append);
    }

private:
    /**
     * @brief Create NPY header for given dtype and shape
     * @param dtype NumPy dtype string (e.g., "<f8", "<i4", "<u4")
     * @param shape_str Shape string (e.g., "(100,)", "(10, 20)")
     * @return Complete header with padding
     */
    static std::string create_header(const std::string &dtype, const std::string &shape_str)
    {
        // Create header dict
        std::ostringstream header_dict;
        header_dict << "{'descr': '" << dtype << "', 'fortran_order': False, 'shape': " << shape_str << ", }";

        std::string header = header_dict.str();

        // Header length must be (total_size - 10) % 64 == 0, where 10 is magic + version + header_len
        // So we pad to make (10 + header_len + padding) % 64 == 0
        size_t base_len = 10; // magic(6) + version(2) + header_len(2)
        size_t current_len = base_len + header.size() + 1; // +1 for newline
        size_t padding = (64 - (current_len % 64)) % 64;

        // Add spaces for padding and newline
        header += std::string(padding, ' ');
        header += '\n';

        return header;
    }

    /**
     * @brief Write NPY file header
     * @param file Output file stream
     * @param dtype NumPy dtype string
     * @param shape_str Shape string
     */
    static void write_header(std::ofstream &file, const std::string &dtype, const std::string &shape_str)
    {
        std::string header = create_header(dtype, shape_str);
        uint16_t header_len = static_cast<uint16_t>(header.size());

        // Write magic string
        file.write("\x93NUMPY", 6);

        // Write version (1.0)
        file.put(1);
        file.put(0);

        // Write header length (little-endian)
        file.write(reinterpret_cast<const char *>(&header_len), 2);

        // Write header
        file.write(header.c_str(), header_len);
    }

    /**
     * @brief Implementation for 1D double array
     */
    static void write_array_1d_impl(const std::string &filename, const double *data, size_t size, bool append)
    {
        std::ios_base::openmode mode = std::ios::binary;
        if (append)
            mode |= std::ios::app;
        else
            mode |= std::ios::trunc;

        std::ofstream file(filename, mode);
        if (!file)
        {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        if (!append)
        {
            // Write header for 1D array of doubles
            std::ostringstream shape;
            shape << "(" << size << ",)";
            write_header(file, "<f8", shape.str()); // <f8 = little-endian float64
        }

        // Write data
        file.write(reinterpret_cast<const char *>(data), size * sizeof(double));
        file.close();
    }

    /**
     * @brief Implementation for 1D int array
     */
    static void write_array_1d_int_impl(const std::string &filename, const int *data, size_t size, bool append)
    {
        std::ios_base::openmode mode = std::ios::binary;
        if (append)
            mode |= std::ios::app;
        else
            mode |= std::ios::trunc;

        std::ofstream file(filename, mode);
        if (!file)
        {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        if (!append)
        {
            std::ostringstream shape;
            shape << "(" << size << ",)";
            write_header(file, "<i4", shape.str()); // <i4 = little-endian int32
        }

        // Write data
        file.write(reinterpret_cast<const char *>(data), size * sizeof(int));
        file.close();
    }

    /**
     * @brief Implementation for 1D uint array
     */
    static void write_array_1d_uint_impl(const std::string &filename, const unsigned int *data, size_t size, bool append)
    {
        std::ios_base::openmode mode = std::ios::binary;
        if (append)
            mode |= std::ios::app;
        else
            mode |= std::ios::trunc;

        std::ofstream file(filename, mode);
        if (!file)
        {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        if (!append)
        {
            std::ostringstream shape;
            shape << "(" << size << ",)";
            write_header(file, "<u4", shape.str()); // <u4 = little-endian uint32
        }

        // Write data
        file.write(reinterpret_cast<const char *>(data), size * sizeof(unsigned int));
        file.close();
    }

    /**
     * @brief Implementation for 2D double array
     */
    static void write_array_2d_impl(const std::string &filename, const double *data,
                                     size_t rows, size_t cols, bool append)
    {
        std::ios_base::openmode mode = std::ios::binary;
        if (append)
            mode |= std::ios::app;
        else
            mode |= std::ios::trunc;

        std::ofstream file(filename, mode);
        if (!file)
        {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        if (!append)
        {
            std::ostringstream shape;
            shape << "(" << rows << ", " << cols << ")";
            write_header(file, "<f8", shape.str());
        }

        // Write data (already in row-major order)
        file.write(reinterpret_cast<const char *>(data), rows * cols * sizeof(double));
        file.close();
    }

    /**
     * @brief Implementation for 2D int array
     */
    static void write_array_2d_int_impl(const std::string &filename, const int *data,
                                         size_t rows, size_t cols, bool append)
    {
        std::ios_base::openmode mode = std::ios::binary;
        if (append)
            mode |= std::ios::app;
        else
            mode |= std::ios::trunc;

        std::ofstream file(filename, mode);
        if (!file)
        {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        if (!append)
        {
            std::ostringstream shape;
            shape << "(" << rows << ", " << cols << ")";
            write_header(file, "<i4", shape.str());
        }

        // Write data
        file.write(reinterpret_cast<const char *>(data), rows * cols * sizeof(int));
        file.close();
    }
};

#endif // NPY_WRITER_HPP
