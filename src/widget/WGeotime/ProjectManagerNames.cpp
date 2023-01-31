#include <QDebug>

#include <ProjectManagerNames.h>

ProjectManagerNames::ProjectManagerNames()
{
	this->tiny.clear();
	this->full.clear();
	this->color.clear();
}

ProjectManagerNames::~ProjectManagerNames()
{
}


bool ProjectManagerNames::isEmpty() const
{
	return full.empty();
}

int ProjectManagerNames::getSize() const
{
	return full.size();
}

const std::vector<QString>& ProjectManagerNames::getTiny() const
{
	return this->tiny;
}

const std::vector<QString>& ProjectManagerNames::getFull() const
{
	return this->full;
}

const std::vector<QBrush>& ProjectManagerNames::getColor() const
{
	return this->color;
}

void ProjectManagerNames::clear()
{
	tiny.clear();
	full.clear();
	color.clear();
}

void ProjectManagerNames::copy(std::vector<QString> _tiny, std::vector<QString> _full)
{
	int N = _tiny.size();
	tiny.resize(N);
	for (int n=0; n<N; n++)
	{
		if ( _tiny[n].isEmpty() ) tiny[n] = getLastFilename(_full[n]); else tiny[n] = _tiny[n];
	}
	N = _full.size();
	full.resize(N);
	for (int n=0; n<N; n++)
	{
		full[n] = _full[n];
	}
	color.clear();
}

void ProjectManagerNames::copy(std::vector<QString> _tiny, std::vector<QString> _full, std::vector<QBrush> _color)
{
	int N = _tiny.size();
	tiny.resize(N);
	for (int n=0; n<N; n++)
	{
		if ( _tiny[n].isEmpty() ) tiny[n] = getLastFilename(_full[n]); else tiny[n] = _tiny[n];
	}
	N = _full.size();
	full.resize(N);
	for (int n=0; n<N; n++)
	{
		full[n] = _full[n];
	}
	N = _color.size();
	color.resize(N);
	for (int n=0; n<N; n++)
	{
		color[n] = _color[n];
	}
}

void ProjectManagerNames::copy(std::vector<QString> _tiny, std::vector<QString> _full, std::vector<QBrush> _color, std::vector<int> _dimx, std::vector<int> _dimy, std::vector<int> _dimz)
{
	int N = _tiny.size();
	tiny.resize(N);
	for (int n=0; n<N; n++)
	{
		if ( _tiny[n].isEmpty() ) tiny[n] = getLastFilename(_full[n]); else tiny[n] = _tiny[n];
	}
	N = _full.size();
	full.resize(N);
	for (int n=0; n<N; n++)
	{
		full[n] = _full[n];
	}
	N = _color.size();
	color.resize(N);
	for (int n=0; n<N; n++)
	{
		color[n] = _color[n];
	}
	N = _dimx.size();
	dimx.resize(N);
	for (int n=0; n<N; n++)
	{
		dimx[n] = _dimx[n];
	}
	N = _dimy.size();
	dimy.resize(N);
	for (int n=0; n<N; n++)
	{
		dimy[n] = _dimy[n];
	}
	N = _dimz.size();
	dimz.resize(N);
	for (int n=0; n<N; n++)
	{
		dimz[n] = _dimz[n];
	}
}

void ProjectManagerNames::add(std::vector<QString> _tiny, std::vector<QString> _full)
{
	int N = _tiny.size();
	for (int n=0; n<N; n++)
	{
		tiny.push_back(_tiny[n]);
	}
	N = _full.size();
	for (int n=0; n<N; n++)
	{
		full.push_back(_full[n]);
	}
}

void ProjectManagerNames::add(std::vector<QString> _tiny, std::vector<QString> _full, std::vector<QBrush> _color)
{
	add(_tiny, _full);
	int N = _color.size();
	for (int n=0; n<N; n++)
	{
		color.push_back(_color[n]);
	}
}



bool ProjectManagerNames::isTextInside(QString str, QString prefix)
{
	QStringList list1 = prefix.split(" ", Qt::SkipEmptyParts);
	int nbsearch = list1.size();
	if ( nbsearch == 0 ) return true;
	int val = 0;
	for ( int s=0; s<nbsearch; s++)
	{
		int idx = str.indexOf(list1[s], 0, Qt::CaseInsensitive);
		if ( idx >= 0 || prefix.isEmpty() ) val ++;
	}
	if ( val > 0 ) return true;
	return false;
}


std::vector<QString> ProjectManagerNames::getNamesFromFullPath(std::vector<QString> in)
{
	int N = in.size();
	std::vector<QString> out;
	out.resize(N);
	for (int n=0; n<N; n++)
	{
		QString tmp = in[n];
		int last = tmp.lastIndexOf("/");
		if ( last == tmp.size()-1 )
		{
			tmp = tmp.left(tmp.size()-1);
			last = tmp.lastIndexOf("/");
		}
		out[n] = tmp.right(tmp.size()-1-last);
	}
	return out;
}


QString ProjectManagerNames::getKeyFromFilename(QString filename, QString key)
{
	QString ret = "";
	char fullkey[10000], buff[10000];
	sprintf(fullkey, "%s%%s\\n", key.toStdString().c_str());
	FILE *pfile = fopen(filename.toStdString().c_str(), "r");
	if ( pfile != NULL )
	{
		int cont = 1, n = 0;
		while ( cont )
		{
			int val = fscanf(pfile, fullkey, buff);
			if ( val > 0 )
			{
				cont = 0;
				ret = QString(buff);
			}
			else
			{
				fgets(buff, 10000, pfile);
			}
			n++;
			if ( n > 10 ) cont = 0;
		}
		fclose(pfile);
	}
	return ret;
}

QString ProjectManagerNames::getKeyTabFromFilename(QString filename, QString key)
{
	QString ret = "";
	char fullkey[10000], buff[10000];
	sprintf(fullkey, "%s\t%%s\\n", key.toStdString().c_str());
	FILE *pfile = fopen(filename.toStdString().c_str(), "r");
	if ( pfile != NULL )
	{
		int cont = 1, n = 0;
		while ( cont )
		{
			int val = fscanf(pfile, fullkey, buff);
			if ( val > 0 )
			{
				cont = 0;
				ret = QString(buff);
			}
			else if ( val == EOF )
			{
				// detect end of file instead of stopping at a fixed number of lines
				cont = 0;
			}
			else
			{
				fgets(buff, 10000, pfile);
			}
			n++;
			// no longer stop at a fixed number of lines
			//if ( n > 10 ) cont = 0;
		}
		fclose(pfile);
	}
	return ret;
}





bool ProjectManagerNames::isMultiKeyInside(QString str, QString key)
{
	if ( key.isEmpty() ) return true;
	QStringList list1 = key.split(" ", Qt::SkipEmptyParts);
    int nbsearch = list1.size();

    int val = 0;
    for (int s=0; s<nbsearch; s++)
    {
    	int idx = str.indexOf(list1[s], 0, Qt::CaseInsensitive);
    	if ( idx >=0 ) val++;
    }
    if ( val == nbsearch || nbsearch == 0)
    {
    	return true;
    }
    else
    {
    	return false;
    }
}

int ProjectManagerNames::getIndexFromVectorString(std::vector<QString> list, QString txt)
{
    for (int i=0; i<list.size(); i++)
    {
        if ( list[i].compare(txt) == 0 )
            return i;
    }
    return -1;
}

QString ProjectManagerNames::removeLastSuffix(QString name)
{
	int lastPoint = name.lastIndexOf(".");
	QString nameNoExt = name.left(lastPoint);
	// QString ext = filename.right(filename.size()-lastPoint-1);
	return nameNoExt;
}

QFileInfoList ProjectManagerNames::getDirectoryList(QString path)
{
    QDir dir(path);
    dir.setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
    dir.setSorting(QDir::Name);
    QFileInfoList list = dir.entryInfoList();
    return list;
}

QString ProjectManagerNames::getLastFilename(QString path)
{
	QFileInfo fi(path);
	return fi.baseName();
}


QString ProjectManagerNames::getAbsolutePath(QString filename)
{
	QFileInfo fi(filename);
	return fi.absolutePath();
}


