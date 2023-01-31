#ifndef IData_H
#define IData_H

#include <QObject>
#include <QList>
#include <QUuid>
#include "viewutils.h"

#include <map>

class IGraphicRepFactory;
class WorkingSetManager;

class IData : public QObject {
	Q_OBJECT
public:
	IData(WorkingSetManager *manager, QObject* parent=0);
	virtual ~IData();

	WorkingSetManager* workingSetManager() const;

	virtual QString name() const =0;
	virtual QUuid dataID() const =0;
	virtual IGraphicRepFactory* graphicRepFactory()=0;

	bool displayPreference(ViewType viewType) const;
	bool displayPreferences(const std::vector<ViewType>& viewTypes) const;
	void setDisplayPreference(ViewType viewType, bool val);
	void setDisplayPreferences(const std::vector<ViewType>& viewTypes, bool val);
	void setAllDisplayPreference(bool val);

signals:
	void displayPreferenceChanged(std::vector<ViewType>, bool);

private:
	WorkingSetManager *m_manager;
	std::map<ViewType, bool> m_displayPreferences;
};

Q_DECLARE_METATYPE(IData*)

#endif
