/*
 * MtLengthUnit.h
 *
 *  Created on: 5 mars 2019
 *      Author: l0222891
 *
 *  From TarumApp, 22 mars 2022
 */

#ifndef NEXTVISION_SRC_UNITS_MTLENGTHUNIT_H_
#define NEXTVISION_SRC_UNITS_MTLENGTHUNIT_H_

#include <QString>
#include <QObject>

class QQmlEngine;
class QJSEngine;

// TODO add like in TarumApp the prefixes (kilo, deci, centi, milli, micro)
class MtLengthUnit {
public:
	static const MtLengthUnit METRE;
	static const MtLengthUnit FEET;

	static const double FEET_TO_METER_RATIO;

	MtLengthUnit(QString name, QString symbol);
	virtual ~MtLengthUnit();

	bool operator==(const MtLengthUnit& other) const;
	bool operator!=(const MtLengthUnit& other) const;

	const QString getSymbol() const { return m_symbol; };
	const QString getName() const { return m_name; };

	static const MtLengthUnit& fromModelUnit(const QString& symbol);
	static double convert(const MtLengthUnit& inUnit, const MtLengthUnit& outUnit, double value);

private:
	QString m_name;
	QString m_symbol;
};

class MtLengthUnitWrapperQML : public QObject {
	Q_OBJECT
public:
	MtLengthUnitWrapperQML(QString name, QString symbol, QObject* parent=nullptr);
	MtLengthUnitWrapperQML(const MtLengthUnit* lengthUnit, QObject* parent=nullptr);
	MtLengthUnitWrapperQML(const MtLengthUnitWrapperQML& lengthUnit, QObject* parent=nullptr);
	MtLengthUnitWrapperQML(QObject* parent=nullptr);
	virtual ~MtLengthUnitWrapperQML();

	const MtLengthUnitWrapperQML& operator=(const MtLengthUnitWrapperQML& other);
	bool operator==(const MtLengthUnitWrapperQML& other) const;
	bool operator!=(const MtLengthUnitWrapperQML& other) const;

	Q_INVOKABLE bool isValid() const;
	Q_INVOKABLE QString getSymbol() const;
	Q_INVOKABLE QString getName() const;

	Q_INVOKABLE double convertFrom(const MtLengthUnitWrapperQML& inUnit, double value);

	static MtLengthUnitWrapperQML METRE;
	static MtLengthUnitWrapperQML FEET;

	static MtLengthUnitWrapperQML* getMetre(QQmlEngine *engine, QJSEngine *scriptEngine);
	static MtLengthUnitWrapperQML* getFeet(QQmlEngine *engine, QJSEngine *scriptEngine);

private:
	const MtLengthUnit* m_object = nullptr;
	bool m_ownPointer = false;
	bool m_valid = false;
};

Q_DECLARE_METATYPE(MtLengthUnitWrapperQML)
Q_DECLARE_METATYPE(const MtLengthUnitWrapperQML*)


#endif /* NEXTVISION_SRC_UNITS_MTLENGTHUNIT_H_ */
