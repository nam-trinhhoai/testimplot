#include "imageformats.h"
#include <regex>

struct Matcher {
	ImageFormats::QSampleType::E e;
    std::vector<std::string> submatchers;
    Matcher(ImageFormats::QSampleType::E e, std::vector<std::string> submatchers) {
        this->e = e;
        this->submatchers = submatchers;
    }
};

static const std::vector<Matcher> MATCHERS = {
    Matcher(ImageFormats::QSampleType::E::UINT8, {R"(uint8)", R"(char)"}),
    Matcher(ImageFormats::QSampleType::E::INT8, {R"(int8)"}),
    Matcher(ImageFormats::QSampleType::E::UINT16,
            {R"(uint16)", R"(unsigned[^[:alnum:]]+short)", R"(short[^[:alnum:]]+unsigned)"}),
    Matcher(ImageFormats::QSampleType::E::INT16, {R"(int16)", R"(short)"}),
    Matcher(ImageFormats::QSampleType::E::UINT32,
            {R"(uint32)", R"(unsigned[^[:alnum:]]+int)", R"(int[^[:alnum:]]+unsigned)"}),
    Matcher(ImageFormats::QSampleType::E::UINT64,
            {R"(uint64)", R"(unsigned[^[:alnum:]]+long)", R"(long[^[:alnum:]]+unsigned)"}),
    Matcher(ImageFormats::QSampleType::E::INT64, {R"(long)", R"(int64)"}),
    Matcher(ImageFormats::QSampleType::E::INT32, {R"(int)", R"(int32)"}),
    Matcher(ImageFormats::QSampleType::E::FLOAT64, {R"(float64)", R"(double)"}),
    Matcher(ImageFormats::QSampleType::E::FLOAT32, {R"(float32)", R"(float)"}) //
};

ImageFormats::QSampleType ImageFormats::QSampleType::find(const std::string &str) {
    for (Matcher matcher : MATCHERS) {
        for (std::string &submatch : matcher.submatchers) {
            std::smatch match;
            std::regex typeMatcher(submatch);
            if (std::regex_search(str, match, typeMatcher)) {
                return matcher.e;
            }
        }
    }
    return QSampleType::E::ERR;
}

