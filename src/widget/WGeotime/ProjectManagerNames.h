
#ifndef __PROJECTMANAGERNAMES__
#define __PROJECTMANAGERNAMES__


#include <vector>
#include <QString>
#include <QBrush>
#include <QDir>


class ProjectManagerNames
{
	public:
		std::vector<QString> tiny;
		std::vector<QString> full;
		std::vector<QBrush> color;
		std::vector<int> dimx;
		std::vector<int> dimy;
		std::vector<int> dimz;
		ProjectManagerNames();
		~ProjectManagerNames();
		bool isEmpty() const;
		int getSize() const;
		void copy(std::vector<QString> _tiny, std::vector<QString> _full);
		void copy(std::vector<QString> _tiny, std::vector<QString> _full, std::vector<QBrush> _color);
		void copy(std::vector<QString> _tiny, std::vector<QString> _full, std::vector<QBrush> _color, std::vector<int> _dimx, std::vector<int> _dimy, std::vector<int> _dimz);
		void add(std::vector<QString> _tiny, std::vector<QString> _full);
		void add(std::vector<QString> _tiny, std::vector<QString> _full, std::vector<QBrush> _color);
		void clear();
		const std::vector<QString>& getTiny() const;
		const std::vector<QString>& getFull() const;
		const std::vector<QBrush>& getColor() const;
		static bool isTextInside(QString str, QString occ);
		static std::vector<QString> getNamesFromFullPath(std::vector<QString> in);
		static QString getKeyFromFilename(QString filename, QString key);
		static QString getKeyTabFromFilename(QString filename, QString key);
		static bool isMultiKeyInside(QString str, QString key);
		static int getIndexFromVectorString(std::vector<QString> list, QString txt);
		static QString removeLastSuffix(QString name);
		static QFileInfoList getDirectoryList(QString path);
		QString getLastFilename(QString path);
		static QString getAbsolutePath(QString filename);

};









#endif
