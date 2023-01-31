#ifndef SUBGRIDGETTERDIALOG_H_
#define SUBGRIDGETTERDIALOG_H_

#include <QDialog>

class QLabel;
class QDialogButtonBox;

class SubGridGetterDialog : public QDialog {
public:
	SubGridGetterDialog(long begin, long end, long step, QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~SubGridGetterDialog();

	void activateMemoryCost(long long oneStepCost);// in bytes

	long outBegin() const;
	long outEnd() const;
	long outStep() const;

private:
	void updateMemoryCost();

	long m_oriBegin;
	long m_oriEnd;
	long m_oriStep;

	long m_outBegin;
	long m_outEnd;
	long m_outStep;

	bool m_activateMemoryCost;
	long long m_oneStepMemoryCost; // in bytes

	QLabel* m_memoryCostFormLabel;
	QLabel* m_memoryCostLabel;
	QLabel* m_availableMemFormLabel;
	QLabel* m_availableMemLabel;
	QDialogButtonBox* m_buttonBox;
};

#endif
