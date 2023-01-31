#ifndef EDITINGSPINBOX_H_
#define EDITINGSPINBOX_H_

#include <QSpinBox>

/**
  * SpinBox that throw signal if editingFinished or steps update value
  *
  * valueChanged does an update when the text changed even when we want to write a multi-digits numbers
  */
class EditingSpinBox : public QSpinBox {
	Q_OBJECT
public:
	EditingSpinBox(QWidget* parent=nullptr);
	~EditingSpinBox();

	virtual void stepBy(int steps) override;

signals:
	void contentUpdated();

private slots:
	void editingFinishedRelay();
};

#endif
