#ifndef NPY_WRITER_HPP
#define NPY_WRITER_HPP

#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <sstream>
#include <iomanip>

/**
 * Utility class for writing NumPy .npy format files
 *
 * Implements NumPy .npy format version 1.0 specification:
 *   Magic string: "\x93NUMPY"
 *   Version: 1.0
 *   Header: Python dict with dtype, shape, fortran_order
 *   Data: Binary array data in C order (row-major)
 */
class NPYWriter
{
public:
    static void write_array_1d(const std::string &filename, const std::vector<double> &data, bool append = false)
    {
        write_array_1d_impl(filename, data.data(), data.size(), append);
    }

    static void write_array_1d(const std::string &filename, const std::vector<int> &data, bool append = false)
    {
        write_array_1d_int_impl(filename, data.data(), data.size(), append);
    }

    static void write_array_1d(const std::string &filename, const std::vector<unsigned int> &data, bool append = false)
    {
        write_array_1d_uint_impl(filename, data.data(), data.size(), append);
    }

    static void write_array_2d(const std::string &filename, const std::vector<double> &data,
                                size_t rows, size_t cols, bool append = false)
    {
        write_array_2d_impl(filename, data.data(), rows, cols, append);
    }

    static void write_array_2d(const std::string &filename, const std::vector<int> &data,
                                size_t rows, size_t cols, bool append = false)
    {
        write_array_2d_int_impl(filename, data.data(), rows, cols, append);
    }

private:
    static std::string create_header(const std::string &dtype, const std::string &shape_str)
    {
        // Create header dict
        std::ostringstream header_dict;
        header_dict << "{'descr': '" << dtype << "', 'fortran_order': False, 'shape': " << shape_str << ", }";

        std::string header = header_dict.str();

        // make (10 + header_len + padding) % 64 == 0
        size_t base_len = 10;
        size_t current_len = base_len + header.size() + 1;
        size_t padding = (64 - (current_len % 64)) % 64;

        // Add spaces for padding and newline
        header += std::string(padding, ' ');
        header += '\n';

        return header;
    }

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
