from itertools import combinations
from server.models.writer import *
from server.views.writersvo import *
from server.views.profilevo import *
from server.models.features import *
from server.httpresponses.errors import *
from server.httpresponses.messages import *
from faker import Faker
import re
import datetime
import copy



def writer_to_dict(writer):
    """
    Convert writer object into dictionary
    :param writer: writer model
    :return: writer_dict: dictionary containing writer's info
    """
    writer_features_dict = writer.features.__dict__
    features = {'_features': writer_features_dict}
    writer_dict = copy.deepcopy( writer.__dict__)
    writer_dict.update(features)

    return writer_dict


def dict_to_writer(writer_dict):
    """
    Convert dictionary to writer object
    :param writer_dict:
    :return: writer model (object)
    """
    writer = Writer()
    writer.id = writer_dict["_id"]
    writer.name = writer_dict["_name"]
    writer.username = writer_dict["_username"]
    writer.image = writer_dict["_image"]
    writer.address = writer_dict["_address"]
    writer.phone = writer_dict["_phone"]
    writer.birthday = writer_dict["_birthday"]
    writer.nid = writer_dict["_nid"]

    features_dict = writer_dict["_features"]
    features = Features()
    features.horest_features = features_dict["_horest_features"]
    features.texture_feature = features_dict["_texture_feature"]
    features.sift_SDS = features_dict["_sift_SDS"]
    features.sift_SOH = features_dict["_sift_SOH"]

    writer.features = features

    return writer


def dict_to_profile(writer_dict, host_url):
    """
    Convert writer dictionary into ProfileVo object
    :param writer_dict: dictionary contains writer's attributes
    :return: ProfileVo object
    """
    profile = ProfileVo()
    profile.id = writer_dict["_id"]
    profile.name = writer_dict["_name"]
    profile.username = writer_dict["_username"]
    profile.address = writer_dict["_address"]
    profile.phone = writer_dict["_phone"]
    profile.nid = writer_dict["_nid"]
    profile.image = host_url + "image/writers/" + writer_dict["_image"]
    profile.birthday = writer_dict["_birthday"]

    return profile


def dict_to_writers(writer_dict):
    """
    Convert writer dictionary into writers model
    :param writer_dict: dictionary returned from database
    :return: writer object of writers model
    """
    writer = WritersVo()
    writer.id = writer_dict["_id"]
    writer.name = writer_dict["_name"]
    writer.username = writer_dict["_username"]

    return writer


def validate_writer_request(request):
    """
    Validate the create writer request regarding phone format, and national ID format
    :param request:
    :return: HTTP Error code:
                - 200 for success
                - 400 if validation failed
             HTTP Message:
                - "OK" for success
                - "Phone is invalid" for invalid phone format
                - "National ID is invalid" for invalid National ID format
    """
    phone_pattern = re.compile("(01)[0 1 2 5][0-9]{8}")
    match_phone = phone_pattern.match(request["_phone"])

    nid_pattern = re.compile("(2|3)[0-9][1-9][0-1][1-9][0-3][1-9](01|02|03|04|11|12|13|14|15|16|17|18|19|21|22|23|24|25|26|27|28|29|31|32|33|34|35|88)\d\d\d\d\d")
    match_nid = nid_pattern.match(request["_nid"])

    if match_phone is None:
        return HttpErrors.BADREQUEST, HttpMessages.INVALIDPHONE
    elif match_nid is None:
        return HttpErrors.BADREQUEST, HttpMessages.INVALIDNID
    else:
        return HttpErrors.SUCCESS, HttpMessages.SUCCESS



def fake_data():
    """
    Function to fake writer's data
    :return: names, birthdays, phones, addresses, nid, images
    """
    fake = Faker()

    names = ["Abdul Ahad", "Abdul Ali", "Abdul Alim", "Abdul Azim", "Abu Abdullah", "Abu Hamza", "Ahmed Tijani", "Ali Reza",
             "Aman Ali", "Anisur Rahman", "Azizur Rahman", "Badr al-Din", "Baha' al-Din", "Barkat Ali", "Burhan al-Din", "Fakhr al-Din",
             "Fazl UrRahman", "Fazlul Karim", "Fazlul Haq", "Ghulam Faruq", "Ghiyath al-Din", "Ghulam Mohiuddin", "Habib ElRahman", "Hamid al-Din",
             "Hibat Allah", "Husam ad-Din", "Ikhtiyar al-Din", "Imad al-Din", "Izz al-Din", "Jalal ad-Din", "Jamal ad-Din", "Kamal ad-Din",
             "Lutfur Rahman", "Mizanur Rahman", "Mohammad Taqi", "Nasir al-Din", "Seif ilislam", "Sadr al-Din", "Sddam Hussein",
             "Ahmed Khairy", "Omar Ali", "Ahmed Gamal", "Gamal Saad", "Ahmed Said", "Ezz Farhan", "Mouhab Farhan", "Sherif Ahmed",
             "Eslam Sherif", "Ahmed Sherif", "Mohamed Ahmed", "Khaled Ali", "Ali Shaalan", "Ahmed Youssry", "Karim Hossam", "Omar Nasharty",
             "AbdelRahman Nasser", "Karim ElRashidy", "Ziad Mansour", "Mohamed Salah", "Anas ElShazly", "Hazem Aly",
             "Youssef Maraghy", "Ebram Hossam", "Mohamed Nour", "Mohamed Ossama", "Hussein Hosny", "Ahmed Samy", "Kareem Haggag",
             "Ahmed Hisham", "Omar Nashaat", "Mohamed Yasser", "Ahmed keraidy", "Magdy Hafez", "Waleed Mostafa", "Khaled Hesham",
             "Sara Hassan", "Rayhana Ayman", "Rana Ali", "Youssra Hussein", "Zeinab Khairy",
             "Samar Gamal", "May Ahmed", "Salma Ibrahim", "Hadeer Hossam", "Hanaa Ahmed", "Bisa Dewidar", "Nachwa Ahmed",
             "Nourhan Farhan", "Mariam Farhan", "Noha Ahmed", "Yasmine Sherif", "Ingy Alaa", "Rana Afifi", "Nour Attya",
             "Amani Tarek", "Salma Ahmed", "Iman Fouad", "Nour Yasser", "Farah Mohamed", "Youmna Helmy",

             "Baniti  Maalouf", "Kaikara  Khouri", "Nona  Baba", "Meroo  Nader", "Nour  Rahal",
             "Raneem  Halabi", "Mona  Bitar", "Iman  Kalb", "Maria  Asker", "Shafira  Basara",
             "Bassant  Sayegh", "Mariah  Maalouf", "Talibah  Harb", "Zalika  Bata", "Kamilah  Baz",
             "Galela  Said", "Shesho  Naser", "Tale  Kanaan", "Sohair  Gaber", "Manar  Kalb",
             "Yasmeen  Dagher", "Amal  Almasi", "Iman  Sabbagh", "Amirah  Almasi", "Amirah  Guirguis",
             "Walaa  Hanania", "Nadeen  Totah", "Arwa  Naifeh", "Olabisis  Dagher", "Quibailah  Ganim",

             "Medo  Boutros", "Aly  Naifeh", "Hamed  Gerges", "Mustafa  Halabi", "Barika  Antoun",
             "Darwishi  Essa", "Bebti  Antoun", "Hesso  Masih", "Kareem  Qureshi", "Chigaru  Bazzi",
             "Ishaq  Essa", "Psamtic  Malouf", "Hamadi  Assaf", "Husani  Arian",
             "Thabit  Cham", "Mustafa  Halabi", "Selim  Ghanem", "Nour  Gaber", "Teremun  Harb", "Adham  Mustafa",
             "Achraf  Isa", "Mohamed Ahmed", "Amir  Tuma", "Momo  Al sadat", "Quasshie  Nader", "Waleed  Halabi",
             "Karim  Sleiman", "Psamtic  Harb"
             ]

    addresses = ["36 El Salam St. - El Saada City - Shoubra El Khima - Cairo - Egypt",
                 "Mahdi St., Azbakiya, Cairo",
                 "125 Shoubra St. -Cairo - Egypt",
                 "67 Mohamad El Nadi Street, Cairo Egypt",
                 "12 Sherif Basha El Kbeer ST, Abdeen",
                 "7 Zohyt Esamt St, From Ein Shams St 11131, Cairo",
                 "100-El Moaz Ledin Elah EL Famtamy, EL Gamlia St., Cairo",
                 "16 Iran St, Dokki, Cairo",
                 "9 Merrit St., in front of Egyptian Museum; Tahrir",
                 "17 Beirut, Heliopolis, Cairo",
                 "Back Of Cairo Airport, El-Salam City. Cairo., Cairo",
                 "18 Egyptian Saudi Finance Bank, El-Nouzha St.",
                 "Back Of Cairo Airport, El-Salam City. Cairo., Cairo",
                 "Heliopolis, Cairo, Cairo",
                 "34 Oraby St. - El Nahada Square Saraiat El Maadi -  Cairo - Egypt",
                 "Shbramant - Giza - Egypt",
                 "El-Gezeira El-Wasta, Egypt Tower",
                 "Zalika  Bata",
                 "Elqanto Sq. Elmoski Cairo - 56, Saad Zaghlol St. Portsaid., cairo",
                 "13 Howd El Gabal Industrial Zone - Cairo - Egypt",
                 "7 El Gezawi St. - Industrial Zone - Shoubra El Khima - El Kaliubia - Egypt",
                 "100-El Moaz Ledin Elah EL Famtamy, EL Gamlia St., Cairo",
                 "26 adly st, ciro egypt",
                 "56 Abdel El Khalek Tharwat, Downtown, cairo",
                 "10 Mohamed Wahid St.- From Bigam Road -Shoubra El Khima - El Kaliubia  - Egypt",
                 "15 Ahmed Mohamed Kamal St. Behindegypt Air In Abas EL Akad/Nasr City/Cairo",
                 "115 Manial Street, Cairo, Cairo",
                 "Al Emdad Wa Altamween Bldgs., Al Nasr Road, Cairo",
                 "sakr korish buildings, New Maadi, cairo",
                 "26 adly st, ciro egypt",
                 "100-El Moaz Ledin Elah EL Famtamy, EL Gamlia St., Cairo",
                 "10 Mohamed Wahid St.- From Bigam Road -Shoubra El Khima - El Kaliubia  - Egypt",
                 "24 Menouf Street, Heliopolis",
                 "6 Octobr, Cairo, Cairo",
                 "Mahdi St., Azbakiya, Cairo",
                 "Mohammed Hafiz St., Mohande Cairo, Cairo",
                 "7 El Gezawi St. - Industrial Zone - Shoubra El Khima - El Kaliubia - Egypt",
                 "58 Ahmed Orabi St., Mohandseen, Cairo",
                 "80 Mosadek, St, Cairo",
                 "8 El-Nozha St.; Egyptian Saudi Construction Co. Bldg.",
                 "85 Saudi Egyptian Construction Co. Bldgs. - Behind Qubba Pal",
                 "3 Shaaker El Gendi St.- El Sharabia - Cairo - Egypt",
                 "Saint Fatima, Heliopolis, Cairo",
                 "National Bank of Egypt Bldg 23 ElRoda St, Mamalik Sq.",
                 "10 Mohamed Wahid St.- From Bigam Road -Shoubra El Khima - El Kaliubia  - Egypt",
                 "9 Merrit St., in front of Egyptian Museum; Tahrir",
                 "Mit Halfa - Behind Seliman Baktar -El Kaliuobia - Egypt",
                 "El Nozha Str., Nasr City, Cairo",
                 "36 Ard El Awkaf Hadaek El Kobba, cairo",
                 "Klm. 25 Misr Esmalia Desert Road - El Abbour City - Cairo - Egypt",
                 "11 Ahmed Mokhtar Hegazy St. Manial",
                 "14 Zaki Elmohandess Elnozha Elgedda",
                 "Saint Fatima, Heliopolis, Cairo",
                 "228 Elhorria Street, Alex",
                 "International Road, Abu Yossief, Alex",
                 "56 ismail mehnna",
                 "Eltabia Alex Egypt",
                 "58 Ahmed Hafez St., Sidibesher, alex",
                 "60 Elgaish Road, Elebrahimia",
                 "Alex Agriculture Rd.",
                 "8a EL Mamar EL Gharby Street. Tanta Egypt",
                 "Extension Of el bahr st.",
                 "15 Ahmed Mohamed Kamal St. Behindegypt Air In Abas EL Akad/Nasr City/Cairo",
                 "St. No. 47 Industrial Zone - El AbbasiaCairo - Egypt",
                 "Beside Military Hotel, Cairo, Cairo",
                 "Beer Homos, Elmosky, Cairo",
                 "First of Mit Ghamr Road - El Sanblawin -  Egypt",
                 "Reija Street, Cairo, Cairo",
                 "Nasir City Cairo Egypt",
                 "Maadi, Cairo, Cairo",
                 "79 H Al Nasr Road, Nasr City, Cairo",
                 "National Bank of Egypt Bldg 23 ElRoda St, Mamalik Sq.",
                 "Behind Nile T. V, Mokattam, Cairo",
                 "6 makhlouf St, Dokky Guiza. Apt. No7, Cairo",
                 "110 Kasarat  ElBaladia St. - El Kosarin -El Zawia El Hamra Cairo - Egypt",
                 "115 Manial Street, Cairo, Cairo",
                 "13A Awlad Emara St. -Bigam Road - El Kanater - El Kliuobia - Egypt",
                 "Industrial Area 1, Plot No 8/9, Cairo",
                 "86948 Thomas Centers Apt. 939 Tinaview, CT 85380",
                 "9754 Kimberly River Apt. 962 Lake Alyssaview, NE 71059",
                 "275 Kelly Trafficway Port Josephfurt, MT 26079",
                 "USS Cunningham FPO AE 78580",
                 "06499 Michelle Road Apt. 235 East Alexis, AL 77635",
                 "645 Sosa Crescent Suite 949 New Amandamouth, ND 28889",
                 "67444 Raymond Cliff New Donald, IN 05820",
                 "9912 Martinez Rapid Suite 549 Stewarttown, AK 46258",
                 "38217 Harvey Ford Port Kayla, IL 03107",
                 "68537 Butler Trail Apt. 016 West Alexanderton, KS 82543"
                 "48092 John Islands North Kristine, DE 37085",
                 "20383 Kevin Road Apt. 633 Gregstad, IL 38966",
                 "78816 Timothy Tunnel North Charles, KY 55302",
                 "91406 Joseph Valley West Austinport, NC 78768",
                 "7182 Jasmine Islands North Edward, SC 90975",
                 "878 Rodriguez Village East Bianca, NY 14554",
                 "49526 Gonzalez Junction Suite 563 Smithstad, WI 03226",
                 "9590 Suzanne Fork Suite 385 Masonburgh, NM 86329",
                 "3856 Mitchell Points New Shannon, NV 39368",
                 "1936 Newman Drive Suite 454 Alvarezborough, WV 86823",
                 "234 Jacqueline Tunnel East Chrischester, AZ 08477"
                 "4883 Flores Parkways Scottview, MO 36747",
                 "122 Ramsis Tower Bldg., El Galaa St.",
                 "203 El Geish St.",
                 "Egyptian Museum Cairo",
                 "Soliman Alhalaby Street, Ramsis, Downtown",
                 "3 Hod El Ghafara - Mit Nama - Kaliwb El Kalubia - Egypt",
                 "58 Gamal El Din Dewedar St. - 8th Zone - Nasr City - Cairo   - Egypt",
                 "Reija Street, Cairo, Cairo",
                 "Mohammed Hafiz St., Mohande Cairo, Cairo",
                 "3 Negma St., Heliopolis, Cairo",
                 "Teraat El Mariutia St. - Kerdasa Road Giza - Egypt",
                 "9a El Sakakeni St., Al Daher, Cairo",
                 "6 Mahmoud Hafez St. - Safir Squrea Misr El Gededa - Cairo - Egypt",
                 "6th Floor, Cairo, Cairo",
                 "03 Kamel Moursy St, Heliopolis, Cairo",
                 "56 Abdel El Khalek Tharwat, Downtown, cairo",
                 "23 Ali Mosa St. - From El Trolly St. - From El Kabalat St. Cairo - Egypt",
                 "12 Al-Hossary Street, Agouza (4 Falouga St. ), Cairo",
                 "26 Agaybi Wasef St, Ain Shams, Cairo",
                 "sakr korish buildings, New Maadi, cairo",
                 "8 Hfiz Ramadan, Madinet Nasr, Cairo",
                 "27 El Mostashfa El Yonani St. - Industrial El Abbasia Cairo  - Egypt",
                 "St., Dokii, Cairo",
                 "Gisr El-suez, Pobox 115 Panorama, Cairo",
                 "16 18 Azoz Hassan St. - Industrial Zone Met Halfa - Kaliobia - Egypt",
                 "46 Nasr Road-ramsis Extinsion -cairo-egypt",
                 "El Helmia El Gededa, 6thfloor, Cairo",
                 "3 El Gamea St. - From Kasarat El Baladia St. - El Zawia El Hamra Cairo - Egypt",
                 "Nasr City, Cairo, Cairo",
                 "El-Nozha El-Gedida, Heliopolis, Cairo",
                 "Haret El Yahood, El Gmalia, Cairo",
                 "Alameria, Alzaiton, Cairo",
                 "57 Ahmed Orabi St., Mohandseen, Cairo",
                 "56 Ahmed Orabi St., Mohandseen, Cairo",
                 "17 Road 270, New Maadi",
                 "Klm. 28 Misr Alex. Desert Road -Abo  Rawash - Egypt",
                 "Tahrir",
                 "33 Ahmad Heshmat, Elzamalik, Cairo",
                 "6makhlouf St, Dokky Guiza. Apt. No7, Cairo",
                 "5th Rabiya St., Nasr City, Cairo",
                 "15 May St. No. 62 - Shobra El Khaima El Kaliubia - Egypt",
                 "5, Ali Rouby Str No. 8, Cairo",
                 "26 El Mostashfa El Yonani St. - Industrial El Abbasia Cairo  - Egypt",
                 "Alex, Alexandria, Alex",
                 "247 Horia Street, Sporting, Alex",
                 "Wahran Street",
                 "15 El Amin St, Andalowes El Haram, Giza",
                 "Abou El Houl El Seyahy St., Mashaal",
                 "3 Abnaa El Gharbeya St. Extension Of El Thalathiny El Gadid St., El Talbeya",
                 "1 Hussein Othman St. Off El Safa & El Marwa St., El Tawabeq",
                 "28 El Marwa St.",
                 "5 Shark El Tersana Club St.",
                 "6B Aswan Sq, Mohandeseen, Giza",
                 "6 Basem El Kateb St. Off El Tahrir St.",
                 "139 Faisal St., El Koum El Akhdar",
                 "5 El Thawra Sq.",
                 "4, Amman Square - Dokki, Giza",
                 "55 Shehab St.",
                 "3 El Mahatta St.",
                 "5 El Mahatta St.",
                 "4 El Mahatta St."]


    # for i in range(78, 100):
    #     addresses.append(fake.address())

    birthdays = []
    for i in range(0, 159):
        birthdays.append(fake.date(pattern="%Y-%m-%d", end_datetime=datetime.date(2000, 12, 12)))

    phones = ["01116055544", "01018028105", "01270625965", "01277217139", "01253194815",
              "01231867718", "01015120204", "01025120204", "01002191561", "01012226165",
              "01014890440", "01055351750", "01050723791", "01002176931", "01219310301",
              "01236664582", "01052918385", "01293917862", "01111661540", "01111661541",
              "01111661542", "01111661543", "01111661544", "01111661545", "01111661546",
              "01111616770", "01227689530", "01001725446", "01127289617", "01238814480",
              "01016256751"]

    for i in range(0, 25):
        phones.append(phones[i][0:5] + '1'+ phones[i][6:])
        phones.append(phones[i][0:5] + '2'+ phones[i][6:])
        phones.append(phones[i][0:5] + '3'+ phones[i][6:])
        phones.append(phones[i][0:5] + '4' + phones[i][6:])
        phones.append(phones[i][0:5] + '5' + phones[i][6:])
        phones.append(phones[i][0:5] + '6' + phones[i][6:])

    nid = ["29612342381123"]
    combination = list(combinations(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 5))
    for i in range(0, 158):
        nid.append(nid[0][0:9] + ''.join(combination[i]))

    # images = []
    # for filename in os.listdir("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/uploads/writers/new"):
    #     name = str(uuid.uuid1()) + ".jpg"
    #     src = "C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/uploads/writers/new/" + filename
    #     dst = "C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/uploads/writers/new/" + name
    #     os.rename(src, dst)
    #     images.append(name)

    images = ['5651aaf4-7427-11e9-af38-34e6d76770a4.jpg', '56533164-7427-11e9-9ce5-34e6d76770a4.jpg', '565490b6-7427-11e9-9d37-34e6d76770a4.jpg',
              '56563e24-7427-11e9-81b4-34e6d76770a4.jpg', '5657c486-7427-11e9-a95e-34e6d76770a4.jpg', '5658d5d8-7427-11e9-a029-34e6d76770a4.jpg',
              '565a5c42-7427-11e9-af24-34e6d76770a4.jpg', '565b6d8a-7427-11e9-addf-34e6d76770a4.jpg', '565c7eca-7427-11e9-b180-34e6d76770a4.jpg',
              '565e2c5c-7427-11e9-bc5a-34e6d76770a4.jpg', '565eef6e-7427-11e9-8184-34e6d76770a4.jpg', '566075d4-7427-11e9-85f5-34e6d76770a4.jpg',
              '56624a5a-7427-11e9-a200-34e6d76770a4.jpg', '5663348a-7427-11e9-99ef-34e6d76770a4.jpg', '566493e6-7427-11e9-b18e-34e6d76770a4.jpg',
              '56657e3e-7427-11e9-9514-34e6d76770a4.jpg', '5666415c-7427-11e9-b5f9-34e6d76770a4.jpg', '566752b0-7427-11e9-82f8-34e6d76770a4.jpg',
              '5669271e-7427-11e9-ad25-34e6d76770a4.jpg', '566c5afa-7427-11e9-ac08-34e6d76770a4.jpg', '566de180-7427-11e9-b9cf-34e6d76770a4.jpg',
              '566ecb9c-7427-11e9-b0a5-34e6d76770a4.jpg', '56705202-7427-11e9-a254-34e6d76770a4.jpg', '5671d876-7427-11e9-bb43-34e6d76770a4.jpg',
              '5672c2ac-7427-11e9-96d8-34e6d76770a4.jpg', '5673faf0-7427-11e9-9908-34e6d76770a4.jpg', '5674e52e-7427-11e9-b79c-34e6d76770a4.jpg',
              '56764498-7427-11e9-9a10-34e6d76770a4.jpg', '567755dc-7427-11e9-ad4a-34e6d76770a4.jpg', '56786728-7427-11e9-aa01-34e6d76770a4.jpg',
              '567a14b6-7427-11e9-8787-34e6d76770a4.jpg', '567eced0-7427-11e9-b323-34e6d76770a4.jpg', '5684e87a-7427-11e9-a83d-34e6d76770a4.jpg',
              '5685d2ae-7427-11e9-9ddc-34e6d76770a4.jpg', '5687ce34-7427-11e9-be99-34e6d76770a4.jpg', '568d99be-7427-11e9-a3b8-34e6d76770a4.jpg',
              '5691deda-7427-11e9-94c6-34e6d76770a4.jpg', '5692f01e-7427-11e9-a142-34e6d76770a4.jpg', '56947686-7427-11e9-8cb6-34e6d76770a4.jpg',
              '5695aede-7427-11e9-9b1b-34e6d76770a4.jpg', '5696c026-7427-11e9-ba54-34e6d76770a4.jpg', '5697aa64-7427-11e9-ae9f-34e6d76770a4.jpg',
              '5698e2c6-7427-11e9-bab7-34e6d76770a4.jpg', '569b0548-7427-11e9-9e75-34e6d76770a4.jpg', '569c3d9a-7427-11e9-86e9-34e6d76770a4.jpg',
              '569d27d8-7427-11e9-b589-34e6d76770a4.jpg', '569e1212-7427-11e9-a05d-34e6d76770a4.jpg', '569efc52-7427-11e9-b436-34e6d76770a4.jpg',
              '56a034a4-7427-11e9-aa8a-34e6d76770a4.jpg', '56a1bb08-7427-11e9-ad57-34e6d76770a4.jpg', '56a34174-7427-11e9-ab1a-34e6d76770a4.jpg',
              '56a4a0d0-7427-11e9-b4fc-34e6d76770a4.jpg', '56a5b218-7427-11e9-88b4-34e6d76770a4.jpg', '56a69c52-7427-11e9-8718-34e6d76770a4.jpg',
              '56a78692-7427-11e9-8bb4-34e6d76770a4.jpg', '56a870cc-7427-11e9-8ae8-34e6d76770a4.jpg', '56a95b0a-7427-11e9-adc8-34e6d76770a4.jpg',
              '56aa1e3e-7427-11e9-a5fc-34e6d76770a4.jpg', '56ab0876-7427-11e9-b9f8-34e6d76770a4.jpg', '56ac19c2-7427-11e9-af6c-34e6d76770a4.jpg',
              '56adc730-7427-11e9-82b5-34e6d76770a4.jpg', '56af268c-7427-11e9-b38d-34e6d76770a4.jpg', '56b037dc-7427-11e9-822f-34e6d76770a4.jpg',
              '56b0fb0c-7427-11e9-a956-34e6d76770a4.jpg', '56b5b558-7427-11e9-ab34-34e6d76770a4.jpg', '56b82606-7427-11e9-85b5-34e6d76770a4.jpg',
              '56b9d362-7427-11e9-8d7e-34e6d76770a4.jpg', '56babda6-7427-11e9-9bce-34e6d76770a4.jpg', '56bc4412-7427-11e9-bc56-34e6d76770a4.jpg',
              '56bdca70-7427-11e9-b550-34e6d76770a4.jpg', '56bedbb8-7427-11e9-af0f-34e6d76770a4.jpg', '56c08936-7427-11e9-ae66-34e6d76770a4.jpg',
              '56c1e882-7427-11e9-b759-34e6d76770a4.jpg', '56c395f4-7427-11e9-806b-34e6d76770a4.jpg', '56c4803a-7427-11e9-a2d4-34e6d76770a4.jpg',
              '1b01d7a6-7427-11e9-b49f-34e6d76770a4.jpg', '1b033708-7427-11e9-81b2-34e6d76770a4.jpg', '1b03f992-7427-11e9-80ed-34e6d76770a4.jpg',
              '1b0558e6-7427-11e9-b873-34e6d76770a4.jpg', '1b06df46-7427-11e9-8693-34e6d76770a4.jpg', '1b0817d0-7427-11e9-9f71-34e6d76770a4.jpg',
              '1b0976f8-7427-11e9-a35d-34e6d76770a4.jpg', '1b0bc090-7427-11e9-90a6-34e6d76770a4.jpg', '1b0d1ffe-7427-11e9-98cd-34e6d76770a4.jpg',
              '1b0f1b70-7427-11e9-a2aa-34e6d76770a4.jpg', '1b11da2e-7427-11e9-8742-34e6d76770a4.jpg', '1b129d5e-7427-11e9-b041-34e6d76770a4.jpg',
              '1b136098-7427-11e9-a543-34e6d76770a4.jpg', '1b14e6f8-7427-11e9-abb1-34e6d76770a4.jpg', '1b166d68-7427-11e9-9c40-34e6d76770a4.jpg',
              '1b197a2e-7427-11e9-93b2-34e6d76770a4.jpg', '1b1b4eac-7427-11e9-9f6b-34e6d76770a4.jpg', '1b1cfc1e-7427-11e9-bd9d-34e6d76770a4.jpg',
              '1b1e0d62-7427-11e9-ab2d-34e6d76770a4.jpg', '1b1f1ea8-7427-11e9-aa21-34e6d76770a4.jpg', '1b20f324-7427-11e9-852e-34e6d76770a4.jpg',
              '1b2315b6-7427-11e9-afd4-34e6d76770a4.jpg', '1b249c1c-7427-11e9-8ee1-34e6d76770a4.jpg', '1b25fb74-7427-11e9-9cb0-34e6d76770a4.jpg',
              '1b27a8e4-7427-11e9-a573-34e6d76770a4.jpg', '30cb2f3e-7911-11e9-b659-34e6d76770a4.jpg', '30ccb59e-7911-11e9-bc0f-34e6d76770a4.jpg',
              '30cd9fdc-7911-11e9-9732-34e6d76770a4.jpg', '30ced834-7911-11e9-aacf-34e6d76770a4.jpg', '30cfe974-7911-11e9-aa86-34e6d76770a4.jpg',
              '30d196e8-7911-11e9-a686-34e6d76770a4.jpg', '30d2a830-7911-11e9-a2d5-34e6d76770a4.jpg', '30d3b992-7911-11e9-ae2f-34e6d76770a4.jpg',
              '30d4cac6-7911-11e9-8c66-34e6d76770a4.jpg', '30d5b4fe-7911-11e9-9d15-34e6d76770a4.jpg', '30d7145a-7911-11e9-b44b-34e6d76770a4.jpg',
              '30d9d318-7911-11e9-afef-34e6d76770a4.jpg', '30dae45e-7911-11e9-a6be-34e6d76770a4.jpg', '30dbce9c-7911-11e9-9590-34e6d76770a4.jpg',
              '30dcb8d8-7911-11e9-8964-34e6d76770a4.jpg', '30ddca26-7911-11e9-a41a-34e6d76770a4.jpg', '30e03ac2-7911-11e9-9400-34e6d76770a4.jpg',
              '30e12500-7911-11e9-ab7d-34e6d76770a4.jpg', '30e20f3e-7911-11e9-877f-34e6d76770a4.jpg', '30e34794-7911-11e9-bd28-34e6d76770a4.jpg',
              '30e431d8-7911-11e9-b5f0-34e6d76770a4.jpg', '30e51c10-7911-11e9-8a3b-34e6d76770a4.jpg', '30e6064c-7911-11e9-886d-34e6d76770a4.jpg',
              '30e71792-7911-11e9-b6b8-34e6d76770a4.jpg', '30e828da-7911-11e9-837b-34e6d76770a4.jpg', '30e93a26-7911-11e9-98d7-34e6d76770a4.jpg',
              '30ea245e-7911-11e9-950e-34e6d76770a4.jpg', '30eb35a6-7911-11e9-8095-34e6d76770a4.jpg', '30ec1fee-7911-11e9-8416-34e6d76770a4.jpg',
              '30ed7f46-7911-11e9-9008-34e6d76770a4.jpg', 'b809ffec-7911-11e9-b639-34e6d76770a4.jpg', 'b80aea3a-7911-11e9-a770-34e6d76770a4.jpg',
              'b80bfb76-7911-11e9-ab4b-34e6d76770a4.jpg', 'b80c97ae-7911-11e9-92a9-34e6d76770a4.jpg', 'b80d5acc-7911-11e9-8367-34e6d76770a4.jpg',
              'b80e6c26-7911-11e9-90f9-34e6d76770a4.jpg', 'b80f2f62-7911-11e9-a33d-34e6d76770a4.jpg', 'b810409a-7911-11e9-9868-34e6d76770a4.jpg',
              'b811ee02-7911-11e9-8202-34e6d76770a4.jpg', 'b814108c-7911-11e9-994c-34e6d76770a4.jpg', 'b816cf52-7911-11e9-b393-34e6d76770a4.jpg',
              'b81966f8-7911-11e9-ad94-34e6d76770a4.jpg', 'b81a7846-7911-11e9-b458-34e6d76770a4.jpg', 'b81c73ca-7911-11e9-8027-34e6d76770a4.jpg',
              'b81e213a-7911-11e9-a628-34e6d76770a4.jpg', 'b8201cc2-7911-11e9-8627-34e6d76770a4.jpg', 'b8210706-7911-11e9-bb71-34e6d76770a4.jpg',
              'b8230286-7911-11e9-b0a9-34e6d76770a4.jpg', 'b825e846-7911-11e9-9c9e-34e6d76770a4.jpg', 'b827bcc0-7911-11e9-a92f-34e6d76770a4.jpg',
              'b8296a30-7911-11e9-85f8-34e6d76770a4.jpg', 'b82a7b7e-7911-11e9-af5c-34e6d76770a4.jpg', 'b82bb3e2-7911-11e9-8062-34e6d76770a4.jpg',
              'b82cec28-7911-11e9-87d2-34e6d76770a4.jpg', 'b82e4b86-7911-11e9-97fc-34e6d76770a4.jpg', 'b82f83cc-7911-11e9-9082-34e6d76770a4.jpg',
              'b8309518-7911-11e9-b9f5-34e6d76770a4.jpg', 'b831f470-7911-11e9-9b86-34e6d76770a4.jpg']

    return names, birthdays, phones, addresses, nid, images