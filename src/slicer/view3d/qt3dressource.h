#ifndef Qt3DRessource_H
#define Qt3DRessource_H
#include <Qt3DCore/QEntity>
#include "viewutils.h"
#include "qmlenumwrappers.h"
#include "mtlengthunit.h"

class Qt3DRessource: public Qt3DCore::QEntity {
	Q_OBJECT
	Q_PROPERTY(double zScale READ zScale CONSTANT WRITE setzScale NOTIFY zScaleChanged)
	Q_PROPERTY(QMLEnumWrappers::SampleUnit sectionTypeQML READ sectionTypeQML CONSTANT WRITE setSectionTypeQML NOTIFY sectionTypeQMLChanged)
	Q_PROPERTY(MtLengthUnitWrapperQML* depthLengthUnitQML READ depthLengthUnitQML CONSTANT WRITE setDepthLengthUnitQML NOTIFY depthLengthUnitQMLChanged)
public:
    explicit Qt3DRessource(Qt3DCore::QNode *parent = nullptr);
    ~Qt3DRessource();

    double zScale() const;
    void setzScale(double val);

    // c++
    SampleUnit sectionType() const;
    void setSectionType(SampleUnit val);

    const MtLengthUnit* depthLengthUnit() const;
    void setDepthLengthUnit(const MtLengthUnit* val);

    // qml
    QMLEnumWrappers::SampleUnit sectionTypeQML() const;
	void setSectionTypeQML(QMLEnumWrappers::SampleUnit val);

	MtLengthUnitWrapperQML* depthLengthUnitQML();
	void setDepthLengthUnitQML(MtLengthUnitWrapperQML* val);

signals:
	void zScaleChanged(double value);

	// c++
	void sectionTypeChanged(SampleUnit value);
	void depthLengthUnitChanged(const MtLengthUnit* value);

	//qml
	void sectionTypeQMLChanged(QMLEnumWrappers::SampleUnit value);
	void depthLengthUnitQMLChanged(MtLengthUnitWrapperQML* value);
private:
	double m_zScale;
	QMLEnumWrappers::SampleUnit m_sectionType = QMLEnumWrappers::SampleUnit::NONE;

	// currently, depthLengthUnit is only set by the c++, QML will only read
	// changes will be needed if QML also set depthLengthUnit
	MtLengthUnitWrapperQML m_depthLengthUnitWrapper; // for QML
	const MtLengthUnit* m_depthLengthUnit = nullptr; // for C++
};

#endif
