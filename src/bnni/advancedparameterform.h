#ifndef ADVANCEDPARAMETERFORM_H
#define ADVANCEDPARAMETERFORM_H

#include <QFormLayout>
#include <QVector>
#include <structures.h>

class QLineEdit;
class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;

class AdvancedParameterForm : public QFormLayout
{
public:
    AdvancedParameterForm(QWidget* parent = 0);

};

#endif // ADVANCEDPARAMETERFORM_H
