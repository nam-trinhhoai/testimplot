#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <QString>
#include <QVector>
#include <limits>

class QLineEdit;
class QComboBox;

#define CHECKPOINT_REFERENCE QString("reference")

enum PrecisionType {
    float16=0, float32=1
};

enum Optimizer {
    gradientDescent=0, momentum=1, adam=2
};

enum NeuralNetwork {
    Dense=0,
    Dnn=1,
    Xgboost=2,
    NoNetwork=3
};

enum SeismicPreprocessing {
    SeismicNone=0, SeismicHat=1
};

enum WellPostprocessing {
    WellNone=0, WellFilter=1
};

enum Activation {
    linear=0, sigmoid=1, relu=2, selu=3, leaky_relu=4
};

enum SectionOrientation {
    INLINE=0, XLINE=1, RANDOM=2
};

enum LogPreprocessing {
    LogNone=0, LogLn=1
};

class CubeDimension {
public:
    CubeDimension(int i=0, int j=0, int k=0) {
        m_i = i;
        m_j = j;
        m_k = k;
    }
    CubeDimension(const CubeDimension& other) {
        m_i = other.m_i;
        m_j = other.m_j;
        m_k = other.m_k;
    }
    ~CubeDimension() {}

    int getI() const {
        return m_i;
    }
    int getJ() const {
        return m_j;
    }
    int getK() const {
        return m_k;
    }
private:
    int m_i;
    int m_j;
    int m_k;
};

class CubeStep {
public:
    CubeStep(float i=0, float j=0, float k=0) {
        m_i = i;
        m_j = j;
        m_k = k;
    }
    CubeStep(const CubeStep& other) {
        m_i = other.m_i;
        m_j = other.m_j;
        m_k = other.m_k;
    }
    ~CubeStep() {}

    float getI() const {
        return m_i;
    }
    float getJ() const {
        return m_j;
    }
    float getK() const {
        return m_k;
    }
private:
    float m_i;
    float m_j;
    float m_k;
};

class CubeOrigin {
public:
    CubeOrigin(float i=0, float j=0, float k=0) {
        m_i = i;
        m_j = j;
        m_k = k;
    }
    CubeOrigin(const CubeOrigin& other) {
        m_i = other.m_i;
        m_j = other.m_j;
        m_k = other.m_k;
    }
    ~CubeOrigin() {}

    float getI() const {
        return m_i;
    }
    float getJ() const {
        return m_j;
    }
    float getK() const {
        return m_k;
    }
private:
    float m_i;
    float m_j;
    float m_k;
};

typedef struct Parameter {
    QString name="";
    float min=0;
    float max=1;
    float InputMin=0;
    float InputMax=1;
    float OutputMin=0;
    float OutputMax=1;
} Parameter;

typedef struct LogParameter {
    QString name="";
    float min=0;
    float max=1;
    float InputMin=0;
    float InputMax=1;
    float OutputMin=0;
    float OutputMax=1;
    LogPreprocessing preprocessing = LogNone;
} LogParameter;

typedef struct SeismicForm {
    QLineEdit* lineEditInputMin = nullptr;
    QLineEdit* lineEditInputMax = nullptr;
    QLineEdit* lineEditOutputMin = nullptr;
    QLineEdit* lineEditOutputMax = nullptr;
} SeismicForm;

typedef struct LogForm {
    QLineEdit* lineEditDataMin = nullptr;
    QLineEdit* lineEditDataMax = nullptr;
    QLineEdit* lineEditInputMin = nullptr;
    QLineEdit* lineEditInputMax = nullptr;
    QLineEdit* lineEditOutputMin = nullptr;
    QLineEdit* lineEditOutputMax = nullptr;
    QComboBox* comboboxPreprocessing = nullptr;
} LogForm;

typedef struct LogSample {
    QVector<float> seismicVals;
    QVector<float> logVals;
    double depth = -1;
    double x = -1;
    double y = -1;
    double z = -1;
} LogSample;

typedef struct Range {
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();
} Range;

typedef struct IntRange {
    int min = std::numeric_limits<int>::max();
    int max = std::numeric_limits<int>::lowest();
} IntRange;

 typedef struct Well {
     QString name="";
     QVector<LogSample> samples;
     QVector<Range> dynamic;
     QVector<IntRange> ranges;
     bool active = true;
     bool isVertical = false;
 } Well;

typedef struct DataInfo {
    CubeDimension dim;
    CubeStep step;
    CubeOrigin origin;
} SurveyInfo;

#endif // STRUCTURES_H
