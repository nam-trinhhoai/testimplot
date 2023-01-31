#ifndef SRC_INFORMATIONMANAGER_IINFORMATIONPANELWIDGET_H
#define SRC_INFORMATIONMANAGER_IINFORMATIONPANELWIDGET_H

#include <QWidget>


class IInformationPanelWidget : public QWidget {
	Q_OBJECT
public:
	IInformationPanelWidget(QWidget* parent=nullptr);
	virtual ~IInformationPanelWidget();

	virtual bool saveChanges() = 0;
};

#endif // SRC_INFORMATIONMANAGER_IINFORMATIONPANELWIDGET_H
