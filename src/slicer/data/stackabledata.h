#ifndef SRC_SLICER_DATA_STACKABLEDATA_H
#define SRC_SLICER_DATA_STACKABLEDATA_H

#include <QObject>
#include <QStringList>
#include <QVector2D>

#include <vector>
#include <memory>

/**
 * ISO is supposed to be defined by a RANGE StackClassType
 */
enum class StackType {
	ISO, CHANNEL, NOTYPE
};

/**
 * This type is linked to the AbstractStack interface
 *
 * It allow to determine quickly the SubClass used (Label or Range)
 */
enum class StackClassType {
	LABEL, RANGE, NOTYPE
};

class AbstractStack;

class StackableData {
public:
	StackableData();
	virtual ~StackableData();

	virtual std::vector<StackType> stackTypes() const = 0;
	virtual std::shared_ptr<AbstractStack> stack(StackType type) = 0;
};

class AbstractStack : public QObject {
	Q_OBJECT
public:
	AbstractStack(QObject* parent=nullptr);
	virtual ~AbstractStack();

	virtual long stackCount() const = 0;
	virtual long stackIndex() const = 0;

	virtual StackClassType stackClassType() const = 0;

signals:
	void stackCountChanged(long stackCount);
	void stackIndexChanged(long stackIndex);

public slots:
	virtual void setStackIndex(long stackIndex) = 0;
};

class AbstractLabelStack : public AbstractStack {
public:
	AbstractLabelStack(QObject* parent=nullptr);
	virtual ~AbstractLabelStack();

	virtual QString stackLabel(long labelIndex) const = 0;
	virtual QStringList stackLabels() const = 0;

	virtual StackClassType stackClassType() const override;
};

class AbstractRangeStack : public AbstractStack {
public:
	AbstractRangeStack(QObject* parent=nullptr);
	virtual ~AbstractRangeStack();

	/**
	 * Expect min == range.x() and max == range.y()
	 * It should be min < max and stackStep > 0
	 *
	 * To support reverse cases :
	 * Functions stackValueFromIndex and stackIndexFromValue are here to manage cases of negative step
	 */
	virtual QVector2D stackRange() const = 0;
	virtual double stackStep() const = 0;

	virtual double stackValueFromIndex(long index) const = 0;
	virtual long stackIndexFromValue(double value) const = 0;

	virtual StackClassType stackClassType() const override;
};

#endif
