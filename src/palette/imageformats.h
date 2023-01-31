#ifndef ImageFormats_H_
#define ImageFormats_H_

#include <string>
#include <limits>
#include <algorithm>

#include "issame.h"
#include "stringutil.h"

class ImageFormats {
public:
	enum QColorFormat {
		GRAY,
		RGB_INTERLEAVED,
		RGB_PLANAR,
		RGBA_INTERLEAVED,
		RGBA_PLANAR,
		RGBA_INDEXED
	};

	/*enum QSampleType {
		INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT32
	};

	static int byteSize(QSampleType type) {
		if (type == INT8 || type == UINT8)
			return 1;
		else if (type == INT16 || type == UINT16)
			return 2;
		else
			return 4;
	}*/
	/** Enum class representing the stored voxels type. */
	class QSampleType {
	public:
		enum E {
			INT8,
			UINT8,
			INT16,
			UINT16,
			INT32,
			UINT32,
			INT64,
			UINT64,
			FLOAT32,
			FLOAT64,
			ERR
		};

		QSampleType() :
				_e(ERR) {
		}
		QSampleType(E e) :
				_e(e) {
		}
		QSampleType(const QSampleType & val)
		{
			this->_e=val._e;
		}
		QSampleType & operator= ( const QSampleType & val )
		{
			this->_e=val._e;
			return *this;
		}
		bool operator==(const QSampleType::E & rhs)
		{
		    return _e == rhs;
		}
		bool EqualsTo(const QSampleType::E & rhs)
		{
			return _e == rhs;
		}

		QSampleType(unsigned int bitWidth, bool isSigned, bool isFloat) {
			switch (bitWidth) {
			case 8:
				_e = isSigned ? INT8 : UINT8;
				break;
			case 16:
				_e = isSigned ? INT16 : UINT16;
				break;
			case 32:
				if (isFloat)
					_e = FLOAT32;
				else
					_e = isSigned ? INT32 : UINT32;
				break;
			case 64:
				if (isFloat)
					_e = FLOAT64;
				else
					_e = isSigned ? INT64 : UINT64;
				break;
			default:
				_e = ERR;
			}
		}

		operator E() const {
			return _e;
		}

		QSampleType(const std::string& str) :
				_e(_parse(str)) {
		}

		std::string str() const {
			switch (_e) {
			case INT8:
				return "int8";
			case UINT8:
				return "uint8";
			case INT16:
				return "int16";
			case UINT16:
				return "uint16";
			case INT32:
				return "int32";
			case UINT32:
				return "uint32";
			case INT64:
				return "int64";
			case UINT64:
				return "uint64";
			case FLOAT32:
				return "float32";
			case FLOAT64:
				return "float64";
			case ERR:
				return "err";
			}
			return "err";
		}

		inline size_t byte_size() const {
			switch (_e) {
			case INT8:
				return 1;
			case UINT8:
				return 1;
			case INT16:
				return 2;
			case UINT16:
				return 2;
			case INT32:
				return 4;
			case UINT32:
				return 4;
			case INT64:
				return 8;
			case UINT64:
				return 8;
			case FLOAT32:
				return 4;
			case FLOAT64:
				return 8;
			case ERR:
				return 0;
			}
			return 0;
		}

		inline size_t bit_size() const {
			return 8 * byte_size();
		}

		inline bool isSigned() {
			switch (_e) {
			case INT8:
			case INT16:
			case INT32:
			case INT64:
			case FLOAT32:
			case FLOAT64:
				return true;
			}
			return false;
		}

		inline bool isFloat() {
			switch (_e) {
			case FLOAT32:
			case FLOAT64:
				return true;
			}
			return false;
		}

		template<typename T>
		bool is_template_equivalent() const {
			switch (_e) {
			case INT8:
				return isSameType<T, int8_t>::value;
			case UINT8:
				return isSameType<T, uint8_t>::value;
			case INT16:
				return isSameType<T, int16_t>::value;
			case UINT16:
				return isSameType<T, uint16_t>::value;
			case INT32:
				return isSameType<T, int32_t>::value;
			case UINT32:
				return isSameType<T, uint32_t>::value;
			case INT64:
				return isSameType<T, int64_t>::value;
			case UINT64:
				return isSameType<T, uint64_t>::value;
			case FLOAT32:
				return isSameType<T, float>::value;
			case FLOAT64:
				return isSameType<T, double>::value;
			case ERR:
				return false;
			}
			return false;
		}

		template<typename T>
		static QSampleType fromTemplate() {
			return QSampleType(_fromTemplate<T>());
		}

		template<typename T>
		T max_value() const {
			switch (_e) {
			case INT8:
				return (T) std::numeric_limits<int8_t>::max();
			case UINT8:
				return (T) std::numeric_limits<uint8_t>::max();
			case INT16:
				return (T) std::numeric_limits<int16_t>::max();
			case UINT16:
				return (T) std::numeric_limits<uint16_t>::max();
			case INT32:
				return (T) std::numeric_limits<int32_t>::max();
			case UINT32:
				return (T) std::numeric_limits<uint32_t>::max();
			case INT64:
				return (T) std::numeric_limits<int64_t>::max();
			case UINT64:
				return (T) std::numeric_limits<uint64_t>::max();
			case FLOAT32:
				return (T) std::numeric_limits<float>::max();
			case FLOAT64:
				return (T) std::numeric_limits<double>::max();
			case ERR:
				return (T) 0;
			}
			return (T) 0;
		}

		template<typename T>
		T min_value() const {
			switch (_e) {
			case INT8:
				return (T) std::numeric_limits<int8_t>::min();
			case UINT8:
				return (T) std::numeric_limits<uint8_t>::min();
			case INT16:
				return (T) std::numeric_limits<int16_t>::min();
			case UINT16:
				return (T) std::numeric_limits<uint16_t>::min();
			case INT32:
				return (T) std::numeric_limits<int32_t>::min();
			case UINT32:
				return (T) std::numeric_limits<uint32_t>::min();
			case INT64:
				return (T) std::numeric_limits<int64_t>::min();
			case UINT64:
				return (T) std::numeric_limits<uint64_t>::min();
			case FLOAT32:
				return (T) std::numeric_limits<float>::min();
			case FLOAT64:
				return (T) std::numeric_limits<double>::min();
			case ERR:
				return (T) 0;
			}
			return (T) 0;
		}

		static QSampleType find(const std::string& str); //try to find the type contained in the string
	private:
		E _e;

		static E _parse(const std::string &str) {
			std::string strl = str;
			trim(strl);
			std::transform(str.begin(), str.end(), strl.begin(), ::tolower);
			if (strl == "int8")
				return INT8;
			if (strl == "uint8")
				return UINT8;
			if (strl == "int16")
				return INT16;
			if (strl == "uint16")
				return UINT16;
			if (strl == "int32")
				return INT32;
			if (strl == "uint32")
				return UINT32;
			if (strl == "int64")
				return INT64;
			if (strl == "uint64")
				return UINT64;
			if (strl == "float")
				return FLOAT32;
			if (strl == "float32")
				return FLOAT32;
			if (strl == "double")
				return FLOAT64;
			if (strl == "float64")
				return FLOAT64;
			return ERR;
		}

		template<typename T>
		static E _fromTemplate() {
			if (isSameType<T, int8_t>::value)
				return INT8;
			if (isSameType<T, uint8_t>::value)
				return UINT8;
			if (isSameType<T, int16_t>::value)
				return INT16;
			if (isSameType<T, uint16_t>::value)
				return UINT16;
			if (isSameType<T, int32_t>::value)
				return INT32;
			if (isSameType<T, uint32_t>::value)
				return UINT32;
			if (isSameType<T, int64_t>::value)
				return INT64;
			if (isSameType<T, uint64_t>::value)
				return UINT64;
			if (isSameType<T, float>::value)
				return FLOAT32;
			if (isSameType<T, double>::value)
				return FLOAT64;
			return ERR;
		}
	};

	static int byteSize(QSampleType type) {
		return type.byte_size();
	}

	static int numBands(QColorFormat colorFormat) {
		int numBands = 1;
		if (colorFormat == ImageFormats::QColorFormat::RGB_PLANAR
				|| colorFormat == ImageFormats::QColorFormat::RGB_INTERLEAVED)
			numBands = 3;
		else if (colorFormat == ImageFormats::QColorFormat::RGBA_PLANAR
				|| colorFormat == ImageFormats::QColorFormat::RGBA_INTERLEAVED)
			numBands = 4;
		return numBands;
	}

private:

	ImageFormats() {
	}
};

#endif
