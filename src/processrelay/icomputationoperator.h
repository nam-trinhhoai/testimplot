#ifndef SRC_PROCESSRELAY_ICOMPUTATIONOPERATOR_H
#define SRC_PROCESSRELAY_ICOMPUTATIONOPERATOR_H

#include <QMetaType>
#include <QString>

class IComputationOperator {
public:
	virtual QString name() = 0;
};

Q_DECLARE_METATYPE(IComputationOperator*)

#endif // SRC_PROCESSRELAY_ICOMPUTATIONOPERATOR_H
