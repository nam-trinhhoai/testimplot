


#include <QGroupBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QComboBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QLabel>


#include <orientationParametersWidget.h>

OrientationParametersWidget::OrientationParametersWidget()
{
	QHBoxLayout* mainLayout = new QHBoxLayout(this);

    QHBoxLayout *qhbox_gradient = new  QHBoxLayout;
    QLabel *label_sigmagradient = new QLabel("size gradient:");
   	lineedit_sigmagradient = new QLineEdit("1.0");
   	lineedit_sigmagradient->setFixedWidth(60);
   	label_sigmagradient->setFixedWidth(label_sigmagradient->fontMetrics().boundingRect(label_sigmagradient->text()).width()*1.5);

   	QHBoxLayout *qhbox_tensor = new  QHBoxLayout;
   	QLabel *label_sigmatensor = new QLabel("size orientation:");
   	lineedit_sigmatensor = new QLineEdit("1.5");
   	lineedit_sigmatensor->setFixedWidth(60);
   	label_sigmatensor->setFixedWidth(label_sigmatensor->fontMetrics().boundingRect(label_sigmatensor->text()).width()*1.5);


   	qhbox_gradient->addWidget(label_sigmagradient);
   	qhbox_gradient->addWidget(lineedit_sigmagradient);
   	qhbox_gradient->setAlignment(Qt::AlignLeft);
   	qhbox_tensor->addWidget(label_sigmatensor);
   	qhbox_tensor->addWidget(lineedit_sigmatensor);
   	qhbox_tensor->setAlignment(Qt::AlignLeft);

	mainLayout->addLayout(qhbox_gradient);
	mainLayout->addLayout(qhbox_tensor);
	mainLayout->setAlignment(Qt::AlignLeft);

	setGradient(1.0);
	setTensor(1.5);
}




OrientationParametersWidget::~OrientationParametersWidget()
{

}



void OrientationParametersWidget::setGradient(double val)
{
	int val0 = (int)(2.0*(3.0*val)+1+.5);
	lineedit_sigmagradient->setText(QString::number(val0));
}

void OrientationParametersWidget::setTensor(double val)
{
	int val0 = (int)(2.0*(3.0*val)+1+.5);
	lineedit_sigmatensor->setText(QString::number(val0));
}

double OrientationParametersWidget::getGradient()
{
	int val0 = lineedit_sigmagradient->text().toInt();
	return (double)((double)val0-1.0)/2.0/3.0;
}

double OrientationParametersWidget::getTensor()
{
	int val0 = lineedit_sigmatensor->text().toInt();
	return (double)((double)val0-1.0)/2.0/3.0;
}
