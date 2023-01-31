#ifndef HORIZONANIMPANELWIDGET_H
#define HORIZONANIMPANELWIDGET_H

#include "iinformationpanelwidget.h"

#include <QColor>
#include <QPointer>

class HorizonAnimInformation;

class QComboBox;

class HorizonAnimPanelWidget : public IInformationPanelWidget {
	Q_OBJECT
public:
	HorizonAnimPanelWidget(HorizonAnimInformation* information, QWidget* parent=nullptr);
	virtual ~HorizonAnimPanelWidget();

	virtual bool saveChanges() override;

//	QColor color() const;

public slots:
	void attributChanged(int i);
	//void editColor();
	//void setColor(QColor color);

private:
//	void setButtonColor(const QColor& color);

	QPointer<HorizonAnimInformation> m_information;

	QComboBox* m_comboAttribut;
	//QColor m_color;
	//QPushButton* m_colorButton;
};

#endif // HORIZONANIMPANELWIDGET_H
