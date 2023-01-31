#ifndef SRC_INFORMATIONMANAGER_NEXTVISIONHORIZON_NEXTVISIONHORIZONINFORMATIONPANELWIDGET_H
#define SRC_INFORMATIONMANAGER_NEXTVISIONHORIZON_NEXTVISIONHORIZONINFORMATIONPANELWIDGET_H

#include "iinformationpanelwidget.h"

#include <QColor>
#include <QPointer>
class WorkingSetManager;

class NextvisionHorizonInformation;

class QPushButton;

class NextvisionHorizonInformationPanelWidget : public IInformationPanelWidget {
	Q_OBJECT
public:
	NextvisionHorizonInformationPanelWidget(NextvisionHorizonInformation* information, WorkingSetManager *workingSetManager, QWidget* parent=nullptr);
	virtual ~NextvisionHorizonInformationPanelWidget();

	virtual bool saveChanges() override;

	QColor color() const;

public slots:
	void editColor();
	void setColor(QColor color);

private:
	void setButtonColor(const QColor& color);

	QPointer<NextvisionHorizonInformation> m_information;

	QColor m_color;
	QPushButton* m_colorButton;
	WorkingSetManager *m_workingSetManager = nullptr;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONPANELWIDGET_H
