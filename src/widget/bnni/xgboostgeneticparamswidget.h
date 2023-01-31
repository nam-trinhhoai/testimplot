#ifndef SRC_WIDGET_BNNI_XGBOOSTGENETICPARAMSWIDGET_H
#define SRC_WIDGET_BNNI_XGBOOSTGENETICPARAMSWIDGET_H

#include "xgboostgeneticparams.h"

#include <QGroupBox>
#include <QPointer>

class QSpinBox;
class QDoubleSpinBox;

class XgBoostGeneticParamsWidget : public QGroupBox {
	Q_OBJECT
public:
	XgBoostGeneticParamsWidget(XgBoostGeneticParams& params, const QString& title, QWidget* parent=0);
	~XgBoostGeneticParamsWidget();

private slots:
	void maxDepthChanged(int val);
	void nEstimatorsChanged(int val);
	void learningRateChanged(double val);
	void subsampleChanged(double val);
	void colsampleByTreeChanged(double val);

private:
	QPointer<XgBoostGeneticParams> m_params;

	QSpinBox* m_maxDepthSpinBox;
	QSpinBox* m_nEstimatorsSpinBox;
	QDoubleSpinBox* m_learningRateSpinBox;
	QDoubleSpinBox* m_subsampleSpinBox;
	QDoubleSpinBox* m_colsampleByTreeSpinBox;
};

#endif // SRC_WIDGET_BNNI_XGBOOSTGENETICPARAMSWIDGET_H
